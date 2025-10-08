#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 02 — Guarded decoding with first-token linear probe (Qwen2.5-VL).

- Loads Qwen2.5-VL (3B/7B Instruct).
- Fits a 1-layer logistic probe from Task01 NPZ (returns W, b, var_idx).
- During decoding, gets first-token logits, scores with probe:
    if prob < threshold -> prepend refusal prefix -> generate;
    else -> generate baseline.
- Saves JSONL examples and a CSV summary.

Example:
CUDA_VISIBLE_DEVICES=0,1 python task02.py \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --limit 2000 \
  --vizwiz-npz runs_qwen7b/7b-instruct_vizwiz/7b-instruct_vizwiz_logits_2160.npz \
  --mathvista-npz runs_qwen7b/7b-instruct_mathvista/7b-instruct_mathvista_logits_500.npz
"""

import argparse, json, os, io, csv, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# Try to import LVLM-LP dataset helpers
try:
    from dataset import build_dataset
    from utils.prompt import Prompter
except Exception:
    build_dataset = None
    Prompter = None

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _to_pil(x):
    if x is None: return None
    if isinstance(x, PILImage.Image): return x.convert("RGB")
    if isinstance(x, (str, Path)):
        try: return PILImage.open(x).convert("RGB")
        except Exception: return None
    if isinstance(x, dict):
        b = x.get("bytes"); p = x.get("path")
        if b is not None:
            try: return PILImage.open(io.BytesIO(b)).convert("RGB")
            except Exception: return None
        if p:
            try: return PILImage.open(p).convert("RGB")
            except Exception: return None
    return None


# -----------------------------
# Prompter / datasets
# -----------------------------
def make_prompter_for_dataset(name: str):
    pt = {"vizwiz": "oeh", "mathvista": "math"}.get(name.lower(), "oeh")
    if Prompter is None:
        class _P: 
            def build_prompt(self, q): return q
        return _P()
    try:
        p = Prompter(pt, pt)
    except TypeError:
        p = Prompter(pt)
    try:
        _ = p.build_prompt("test")
    except Exception:
        tmpl = getattr(p, "question_template", "%s")
        p.build_prompt = lambda q, _t=tmpl: _t % q
    return p

def load_vizwiz(split="val", limit=None) -> List[Dict]:
    assert build_dataset is not None, "LVLM-LP dataset builder not found."
    p = make_prompter_for_dataset("vizwiz")
    data, _ = build_dataset("VizWiz", split, p)
    if limit: data = data[:limit]
    return data

def load_mathvista(split="testmini", limit=None) -> List[Dict]:
    if build_dataset is not None:
        p = make_prompter_for_dataset("mathvista")
        data, _ = build_dataset("MathVista", split, p)
        if limit: data = data[:limit]
        return data
    # Fallback (shouldn’t be needed if your repo is set up)
    from datasets import load_dataset
    ds = load_dataset("AI4Math/MathVista", split=split)
    out = []
    for r in ds:
        img = r.get("image") or r.get("img")
        if isinstance(img, str):
            pth = Path("data/MathVista/images")/split/Path(img).name
            img = str(pth) if pth.exists() else img
        q = r.get("question") or r.get("query") or ""
        out.append({"img_path": img, "question": q, "label": 0})
        if limit and len(out) >= limit: break
    return out


# -----------------------------
# Probe fitting from NPZ
# -----------------------------
def fit_probe_from_npz(npz_path: str, topk: int = 4096, solver: str = "saga",
                       val_ratio: float = 0.2):
    """
    Train a logistic probe from Task01 NPZ (expects keys X, y).
    Returns:
      W: [1,K] float32 tensor
      b: [1]   float32 tensor
      var_idx: LongTensor[K] or None (which vocab dims were kept)
      meta: dict
    """
    z = np.load(npz_path)
    X = z["X"]; y = z["y"]
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"Empty/malformed: {npz_path}  (X={X.shape}, y={y.shape})")

    n = len(y)
    n_val = max(1, int(np.ceil(n * val_ratio)))
    Xtr, ytr = X[:-n_val], y[:-n_val]

    var_idx = None
    if topk is not None and topk < X.shape[1]:
        var = Xtr.var(axis=0)
        var_idx = np.argsort(var)[-topk:]
        Xtr = Xtr[:, var_idx]

    clf = LogisticRegression(max_iter=500, solver=solver, n_jobs=64)
    clf.fit(Xtr, ytr)

    W = torch.from_numpy(clf.coef_.astype(np.float32))      # [1,K]
    b = torch.from_numpy(clf.intercept_.astype(np.float32)) # [1]
    var_idx_t = torch.as_tensor(var_idx, dtype=torch.long) if var_idx is not None else None
    meta = {"n": int(n), "dim": int(Xtr.shape[1]), "topk": int(topk) if topk else None}
    return W, b, var_idx_t, meta


@torch.inference_mode()
def score_guard(x_np, W, b, var_idx=None):
    """
    x_np: np.ndarray [V] first-token logits (full vocab)
    W:    [1,K]; b: [1]
    var_idx: optional LongTensor[K]
    """
    x = torch.as_tensor(x_np, dtype=torch.float32).view(1, -1)
    if var_idx is not None:
        x = x.index_select(1, var_idx)  # -> [1,K]
    z = (x @ W.T) + b
    return torch.sigmoid(z).item()


# -----------------------------
# Qwen2.5-VL wrapper
# -----------------------------
class QwenGuard:
    def __init__(self, model_id: str):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", dtype=dtype
        ).eval()

    def _pack(self, pil_image: PILImage.Image, text: str):
        messages = [{"role":"user","content":[{"type":"image","image":pil_image},
                                              {"type":"text","text":text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
        dev = next(self.model.parameters()).device
        for k,v in list(inputs.items()):
            if hasattr(v, "to"): inputs[k] = v.to(dev)
        return inputs

    @torch.inference_mode()
    def first_token_logits(self, pil_image, text) -> np.ndarray:
        inputs = self._pack(pil_image, text)
        out = self.model(**inputs)
        idx = inputs["input_ids"].shape[-1] - 1
        return out.logits[:, idx, :].squeeze(0).float().cpu().numpy()

    @torch.inference_mode()
    def generate(self, pil_image, text, max_new_tokens=128) -> str:
        inputs = self._pack(pil_image, text)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]

    @torch.inference_mode()
    def generate_with_prefix(self, pil_image, text, prefix, max_new_tokens=128) -> str:
        messages = [{"role":"user","content":[{"type":"image","image":pil_image},
                                              {"type":"text","text":text}]}]
        base = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        injected = base + prefix
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[injected], images=image_inputs, videos=video_inputs, return_tensors="pt")
        dev = next(self.model.parameters()).device
        for k,v in list(inputs.items()):
            if hasattr(v, "to"): inputs[k] = v.to(dev)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]


# -----------------------------
# Guarded decoding
# -----------------------------
def make_refusal_prefix(dataset_key: str) -> str:
    key = dataset_key.lower()
    if key == "vizwiz":
        return "Sorry, this image likely does not provide enough information to answer the question. Because "
    if key == "mathvista":
        return "I am uncertain about the answer. Instead, I will explain the reasoning and ambiguities: "
    return "I prefer not to answer directly. Instead, here is a safer response: "

def run_guarded_eval(
    model: QwenGuard,
    dataset_key: str,
    items: List[Dict],
    W: torch.Tensor, b: torch.Tensor, var_idx: Optional[torch.Tensor],
    save_dir: Path,
    limit: Optional[int] = None,
    max_new_tokens: int = 64,
    prob_threshold: float = 0.5,
) -> None:
    save_dir = ensure_dir(save_dir)
    ref_prefix = make_refusal_prefix(dataset_key)
    n = len(items) if limit is None else min(limit, len(items))
    out_jsonl = open(save_dir / f"{dataset_key}_task02_examples.jsonl", "w")
    stats = {"accept": 0, "refuse": 0}

    for i in tqdm(range(n), desc=f"{dataset_key}: guarded decoding"):
        ex = items[i]
        img = _to_pil(ex["img_path"])
        if img is None: continue
        q = ex.get("question", "")
        if not isinstance(q, str): q = str(q)

        baseline = model.generate(img, q, max_new_tokens=max_new_tokens)
        first_logits = model.first_token_logits(img, q)
        p = score_guard(first_logits, W, b, var_idx=var_idx)

        if p < prob_threshold:
            stats["refuse"] += 1
            guarded = model.generate_with_prefix(img, q, ref_prefix, max_new_tokens=max_new_tokens)
            decision = "refuse"
        else:
            stats["accept"] += 1
            guarded = baseline
            decision = "accept"

        rec = {
            "id": i, "dataset": dataset_key, "prob_safe": float(p),
            "decision": decision, "question": q, "baseline": baseline,
            "guarded": guarded, "img_path": ex["img_path"],
        }
        out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_jsonl.close()

    with open(save_dir / f"{dataset_key}_task02_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "total", "accept", "refuse", "threshold"])
        w.writerow([dataset_key, n, stats["accept"], stats["refuse"], prob_threshold])


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--datasets", nargs="+", choices=["vizwiz","mathvista"], default=["vizwiz"])
    ap.add_argument("--vizwiz-npz", type=str, required=False)
    ap.add_argument("--mathvista-npz", type=str, required=False)
    ap.add_argument("--save-dir", type=str, default="runs_task02")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    save_root = ensure_dir(args.save_dir)

    print(f"[init] loading {args.model_id} ...")
    guard = QwenGuard(args.model_id)

    # VIZWIZ
    if "vizwiz" in args.datasets:
        if not args.vizwiz_npz:
            raise ValueError("Provide --vizwiz-npz pointing to Task01 logits.")
        W_vw, b_vw, idx_vw, _ = fit_probe_from_npz(args.vizwiz_npz, topk=args.topk, solver="saga")
        data_vw = load_vizwiz(split="val", limit=None)
        run_guarded_eval(
            guard, "vizwiz", data_vw, W_vw, b_vw, idx_vw,
            save_root / "vizwiz", limit=args.limit, prob_threshold=args.threshold
        )

    # MATHVISTA
    if "mathvista" in args.datasets:
        if not args.mathvista_npz:
            raise ValueError("Provide --mathvista-npz pointing to Task01 logits.")
        W_mv, b_mv, idx_mv, _ = fit_probe_from_npz(args.mathvista_npz, topk=args.topk, solver="saga")
        data_mv = load_mathvista(split="testmini", limit=None)
        run_guarded_eval(
            guard, "mathvista", data_mv, W_mv, b_mv, idx_mv,
            save_root / "mathvista", limit=args.limit, prob_threshold=args.threshold
        )

    print(f"[done] results under: {save_root.resolve()}")


if __name__ == "__main__":
    main()
