#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Task 01 runner for LVLM-LP (VizWiz, MathVista, POPE).

- Uses the repo's dataset/model builders and prompt templates where possible.
- Supports two backends:
    1) 'llava7b'  -> LVLM-LP's model.build_model (paper's original 7B)
    2) 'qwen3b'   -> HuggingFace Qwen2.5-VL-3B-Instruct

Outputs per model+dataset:
- cached features (npz) for first-token logits  (and optional hidden states)
- confusion matrix + ROC curve
- token-ID vs ΔAUC/ΔACC plots (like Fig.3 a,b)
- hidden-layer vs ΔACC plots (like Fig.3 d) when hidden states are available
- a consolidated results.json

Run examples:
 CUDA_VISIBLE_DEVICES=0,1 python unified_task01.py --backend qwen3b --out_dir runs_qwen3b --num_samples 4000 --num_chunks 2
 CUDA_VISIBLE_DEVICES=0,1 python unified_task01.py --backend llava7b --model_path liuhaotian/llava-v1.5-7b --out_dir runs_llava7b --num_samples 4000 --num_chunks 2
"""

import os
import io
import json
import time
import math
import argparse
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PILImage
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)

# ---------------------------------------------------------------------
# Repo imports (available once this script sits in LVLM-LP repo root)
# ---------------------------------------------------------------------
# They provide build_dataset / build_model / prompt helpers we’ll reuse.
# If you moved the script elsewhere, add the repo root to PYTHONPATH.
from dataset import build_dataset              # repo: dataset builder
from model import build_model                  # repo: model builder (LLaVA-7B)
from utils.prompt import Prompter              # repo: consistent prompts
from utils.func import get_chunk               # repo: chunk helper

# ---------------------------------------------------------------------
# HF Qwen2.5-VL backend (used when --backend qwen3b)
# ---------------------------------------------------------------------
from transformers import AutoProcessor, AutoModelForCausalLM

QWEN_3B = "Qwen/Qwen2.5-VL-3B-Instruct"

# -----------------------------
# Utility helpers
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def device_of(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def to_pil(image_any) -> Optional[PILImage.Image]:
    """Robust image loader for paths/PIL/bytes."""
    if image_any is None:
        return None
    if isinstance(image_any, PILImage.Image):
        return image_any.convert("RGB")
    if isinstance(image_any, (str, Path)):
        try:
            return PILImage.open(image_any).convert("RGB")
        except Exception:
            return None
    if isinstance(image_any, dict):
        b = image_any.get("bytes"); p = image_any.get("path")
        if b is not None:
            try:
                return PILImage.open(io.BytesIO(b)).convert("RGB")
            except Exception:
                return None
        if p:
            try:
                return PILImage.open(p).convert("RGB")
            except Exception:
                return None
    return None

# -----------------------------
# Dataset download conveniences
# -----------------------------
def maybe_download_vizwiz(repo_root: Path) -> None:
    """
    Use the repo's expectation for VizWiz (train/val lists + image dir).
    If scripts are present, run them; otherwise we print a one-liner fallback.
    (MathVista downloads via HF automatically in repo code.)
    """
    # The repo README shows dataset helpers; if their scripts exist, call them.
    scripts_dir = repo_root / "scripts" / "dataset"
    shs = ["download_VizWiz.sh", "build_VizWiz_list.sh"]
    present = all((scripts_dir / s).exists() for s in shs)
    if present:
        print("[vizwiz] running repo dataset scripts...")
        os.system(f"bash {scripts_dir/'download_VizWiz.sh'}")
        os.system(f"bash {scripts_dir/'build_VizWiz_list.sh'}")
    else:
        print("[vizwiz] repo dataset scripts not found; ensure images & lists follow LVLM-LP expectations.")
        print("         As a fallback, you can use HF dataset `lmms-lab/VizWiz-VQA` (val split) and map "
              "`category=='unanswerable'` → 0/1 labels consistently.")

def maybe_download_pope(repo_root: Path) -> None:
    scripts_dir = repo_root / "scripts" / "dataset"
    sh = scripts_dir / "download_POPE.sh"
    if sh.exists():
        print("[pope] running repo dataset script...")
        os.system(f"bash {sh}")
    else:
        print("[pope] repo dataset script not found; ensure COCO 2014 images and POPE annotations are placed per README.")

# -----------------------------
# Backend adapters
# -----------------------------
class LLaVA7BAdapter:
    """
    Adapter around the repo's build_model() interface. We call the model's
    forward_with_probs(image, prompt) which returns (response, output_ids, logits, probs)
    and we read the logit row of the first generated token (index 0).
    """
    def __init__(self, model_path: str, temperature: float = 0.2, top_p: Optional[float] = None,
                 num_beams: int = 1, device: Optional[str] = None):
        class Args:
            pass
        args = Args()
        args.model_name   = "LLaVA-7B"
        args.model_path   = model_path
        args.temperature  = temperature
        args.top_p        = top_p
        args.num_beams    = num_beams
        self.model = build_model(args)  # repo function

    def first_token_logits(self, image_pil: PILImage.Image, prompt: str) -> np.ndarray:
        import cv2
        img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        resp, output_ids, logits, probs = self.model.forward_with_probs(img, prompt)
        if logits is None or len(logits) == 0:
            raise RuntimeError("No logits returned.")
        # logits is [seq_len, vocab]; first generated token is index 0
        row0 = logits[0]
        return np.array(row0, dtype=np.float32)

    def hidden_states_first_token(self, image_pil: PILImage.Image, prompt: str) -> Optional[np.ndarray]:
        # only available through _basic_forward in extract_hidden_states.py;
        # for simplicity we skip hidden states on llava backend here.
        return None


from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

QWEN_3B = "Qwen/Qwen2.5-VL-3B-Instruct"

def _model_device(m):
    try:
        return next(m.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cpu")

class Qwen3BAdapter:
    """
    Qwen2.5-VL-3B-Instruct via HF. Builds a chat-style prompt with an <image> token
    and extracts the logits at the first generated position.
    """
    def __init__(self, model_id: str = QWEN_3B):
        import torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # L40s do BF16 very well.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        ).eval()

    def _pack(self, pil_image, user_text: str):
        """
        Build messages -> (text prompt, vision inputs) -> processor() tensors.
        Using qwen-vl-utils.process_vision_info is the canonical way for VL inputs.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        dev = _model_device(self.model)
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(dev)
        return inputs

    def first_token_logits(self, pil_image, user_text: str):
        import torch
        with torch.no_grad():
            inputs = self._pack(pil_image, user_text)
            out = self.model(**inputs)
        # position right before generation starts
        idx = inputs["input_ids"].shape[-1] - 1
        row = out.logits[:, idx, :].squeeze(0)  # [vocab]
        return row.detach().float().cpu().numpy()

    # Optional: collect per-layer hidden states if you need layer curves
    def hidden_states_first_token(self, pil_image, user_text: str):
        import torch
        with torch.no_grad():
            inputs = self._pack(pil_image, user_text)
            out = self.model(**inputs, output_hidden_states=True)
        idx = inputs["input_ids"].shape[-1] - 1
        feats = [hs[:, idx, :].squeeze(0).detach().float().cpu().numpy()
                 for hs in out.hidden_states]
        import numpy as np
        return np.stack(feats, axis=0)  # [n_layers, dim]


# -----------------------------
# Feature extraction and caching
# -----------------------------
def extract_logits_for_dataset(
    dataset: List[Dict],
    prompter: Prompter,
    adapter,           # LLaVA7BAdapter or Qwen3BAdapter
    out_cache: Path,
    max_items: Optional[int] = None,
    desc: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X = first-token logits, y = labels."""
    ensure_dir(out_cache.parent)
    Xs, ys = [], []
    errs = 0
    pbar = tqdm(total=(max_items or len(dataset)), desc=f"extract {desc}")
    for i, ins in enumerate(dataset):
        if max_items is not None and i >= max_items:
            break
        img_path = ins["img_path"]
        label    = int(ins["label"])
        prompt   = ins["question"]
        pil = to_pil(img_path)
        if pil is None:
            errs += 1; pbar.update(1); continue
        try:
            logits = adapter.first_token_logits(pil, prompter(prompt))
            Xs.append(logits); ys.append(label)
        except Exception:
            errs += 1
        pbar.update(1)
    pbar.close()
    X = np.stack(Xs, axis=0) if len(Xs) else np.zeros((0, 0), dtype=np.float32)
    y = np.array(ys, dtype=np.int64)
    np.savez_compressed(out_cache, X=X, y=y)
    print(f"[cache] saved {out_cache.name}: X={X.shape}, y={y.shape}, skipped={errs}")
    return X, y


def extract_hidden_for_dataset(
    dataset: List[Dict],
    prompter: Prompter,
    adapter,           # backend which implements hidden_states_first_token (Qwen3BAdapter)
    out_cache: Path,
    max_items: Optional[int] = None,
    desc: str = "",
) -> Optional[np.ndarray]:
    """Return H with shape [N, n_layers, dim] when available."""
    ensure_dir(out_cache.parent)
    Hs = []
    errs = 0
    pbar = tqdm(total=(max_items or len(dataset)), desc=f"hidden {desc}")
    for i, ins in enumerate(dataset):
        if max_items is not None and i >= max_items:
            break
        pil = to_pil(ins["img_path"])
        if pil is None:
            errs += 1; pbar.update(1); continue
        try:
            H = adapter.hidden_states_first_token(pil, prompter(ins["question"]))
            if H is not None:
                Hs.append(H)
        except Exception:
            errs += 1
        pbar.update(1)
    pbar.close()
    if not Hs:
        print("[hidden] none collected for this backend.")
        return None
    H = np.stack(Hs, axis=0)  # [N, L, D]
    np.savez_compressed(out_cache, H=H)
    print(f"[cache] saved {out_cache.name}: H={H.shape}, skipped={errs}")
    return H


# -----------------------------
# Linear probe + plots
# -----------------------------
def linear_probe(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, solver="saga",
                 topk: Optional[int] = 4096, random_state: int = 42) -> Dict:
    """
    Train logistic regression with (optional) top-k feature sparsification
    for stability and speed on very wide vocab logits.
    """
    assert X.ndim == 2 and len(X) == len(y), "bad shapes for LP"
    n = len(y)
    n_val = int(math.ceil(n * val_ratio))
    Xtr, Xval = X[:-n_val], X[-n_val:]
    ytr, yval = y[:-n_val], y[-n_val:]

    if topk is not None and topk < X.shape[1]:
        # keep topk by global variance as simple filter
        var = Xtr.var(axis=0)
        idx = np.argsort(var)[-topk:]
        Xtr = Xtr[:, idx]; Xval = Xval[:, idx]
    else:
        idx = None

    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, solver=solver, n_jobs=8)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xval)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    acc = float(accuracy_score(yval, pred))
    f1  = float(f1_score(yval, pred))
    try:
        auc = float(roc_auc_score(yval, prob))
    except Exception:
        auc = float("nan")
    t_elapsed = time.time() - t0

    return {"acc": acc, "f1": f1, "auc": auc, "elapsed_s": t_elapsed,
            "n_train": int(len(ytr)), "n_val": int(len(yval)),
            "dim": int(Xtr.shape[1]), "topk": (int(topk) if topk else None),
            "solver": solver, "var_idx": (idx.tolist() if idx is not None else None)}


def plot_confusion_and_roc(y_true: np.ndarray, y_prob: np.ndarray, title_prefix: str, out_dir: Path):
    pred = (y_prob >= 0.5).astype(np.int64)
    cm = confusion_matrix(y_true, pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title(f"{title_prefix} – Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix}_confusion.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1], [0,1], "k--")
    plt.title(f"{title_prefix} – ROC")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix}_roc.png", dpi=160)
    plt.close()


def plot_token_delta(metrics_by_token: List[Tuple[int, float]], ylabel: str, title: str, out_path: Path):
    xs = [k for k,_ in metrics_by_token]; ys = [v for _,v in metrics_by_token]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title); plt.xlabel("Token ID"); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()


def plot_layer_curve(xs: List[int], ys: List[float], ylabel: str, title: str, out_path: Path):
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title); plt.xlabel("Hidden Layer"); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()


# -----------------------------
# Orchestration
# -----------------------------
TASKS = [
    # (dataset_key, split, prompt_theme, short_tag)
    ("VizWiz",   "val",  "unanswerable", "vizwiz"),
    ("MathVista","val",  "math",         "mathvista"),
    ("POPE",     "val",  "hallucination","pope"),
]

def run_task(args, backend_name: str, model, repo_root: Path, out_dir: Path):
    """
    For each dataset:
      1) build data via repo (ensures same labels/splits)
      2) extract first-token logits (+ optional hidden states for Qwen)
      3) fit linear probe & generate plots
      4) optional token/layer ablations (like Fig.3)
    """
    all_results = {}

    # dataset prep (download if needed)
    if any(t[0] == "VizWiz" for t in TASKS):
        maybe_download_vizwiz(repo_root)
    if any(t[0] == "POPE" for t in TASKS):
        maybe_download_pope(repo_root)

    for ds_name, split, theme, tag in TASKS:
        print(f"\n==> Dataset: {ds_name} / split={split} / theme={theme}")
        prompter = Prompter(args.prompt, theme)

        # Build with repo helper so we get consistent {img_path, question, label}
        data, extra_keys = build_dataset(ds_name, split, prompter)

        # chunk across GPUs/processes if requested
        idxs = list(range(len(data)))
        idxs = get_chunk(idxs, args.num_chunks, args.chunk_idx)
        data = [data[i] for i in idxs]
        if args.num_samples:
            data = data[:args.num_samples]

        ds_tag = f"{backend_name}_{tag}"
        task_out = ensure_dir(out_dir / ds_tag)
        cache_logits = task_out / f"{ds_tag}_logits_{len(data)}.npz"
        cache_hidden = task_out / f"{ds_tag}_hidden_{len(data)}.npz"

        # 1) features
        if cache_logits.exists() and not args.overwrite:
            npz = np.load(cache_logits)
            X = npz["X"]; y = npz["y"]
            print(f"[cache] loaded {cache_logits.name}: X={X.shape}, y={y.shape}")
        else:
            X, y = extract_logits_for_dataset(
                data, prompter, model, cache_logits,
                max_items=len(data), desc=f"{ds_tag}"
            )

        # optional hidden states for layer ablations (supported on Qwen backend)
        H = None
        if args.collect_hidden and hasattr(model, "hidden_states_first_token"):
            if cache_hidden.exists() and not args.overwrite:
                npz = np.load(cache_hidden)
                H = npz["H"]
                print(f"[cache] loaded {cache_hidden.name}: H={H.shape}")
            else:
                H = extract_hidden_for_dataset(
                    data, prompter, model, cache_hidden,
                    max_items=min(len(data), args.hidden_max), desc=f"{ds_tag}"
                )

        # 2) linear probe
        if X.size == 0:
            print(f"[warn] no features for {ds_tag}, skipping.")
            continue

        lp = linear_probe(X, y, val_ratio=0.2, solver="saga", topk=args.topk)
        print(f"{ds_tag}: "
              f"train={lp['n_train']}, val={lp['n_val']}, dim={lp['dim']}, "
              f"cfg=sparse(topk={lp['topk']})/{lp['solver']} "
              f"| Acc={lp['acc']:.4f} F1={lp['f1']:.4f} AUC={lp['auc']:.4f} "
              f"| elapsed={lp['elapsed_s']:.2f}s")

        # plot confusion + ROC on the val slice
        n = len(y); n_val = lp["n_val"]
        y_val = y[-n_val:]
        # rebuild val probs using trained clf
        idx = lp.get("var_idx")
        X_val = X[-n_val:, idx] if idx is not None else X[-n_val:]
        clf = LogisticRegression(max_iter=2000, solver=lp["solver"], n_jobs=8)
        clf.fit(X[:-n_val, idx] if idx is not None else X[:-n_val], y[:-n_val])
        y_prob = clf.predict_proba(X_val)[:,1]

        plot_confusion_and_roc(y_val, y_prob, title_prefix=f"{ds_tag}", out_dir=task_out)

        # 3) Token-ID Δ-metrics (rerun a tiny subset at token ids 0..9)
        #    We approximate Δmetric := metric(token_k) - metric(token_0).
        #    For speed we subsample up to ablate_max items.
        if args.token_curve:
            Ks = list(range(10))
            ablate_n = min(len(data), args.ablate_max)
            sub_data = data[:ablate_n]
            # gather logits for each k
            base_probs = None
            deltas_auc = []
            deltas_acc = []
            for k in Ks:
                # For llava backend, we always used first generated token.
                # To emulate token-k, we simply shift the row we pick when the backend
                # returns multiple positions. Qwen backend only returns first-token logits;
                # for brevity we provide the curve only for llava7b (repo) where sequence logits are accessible.
                if isinstance(model, LLaVA7BAdapter):
                    # call the repo's runner model once per sample to get full logits, then slice row k
                    rows = []
                    for ins in sub_data:
                        pil = to_pil(ins["img_path"])
                        if pil is None: continue
                        import cv2
                        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                        _, _, logits, _ = model.model.forward_with_probs(img, prompter(ins["question"]))
                        if logits is None or len(logits) <= k: continue
                        rows.append(np.array(logits[k], dtype=np.float32))
                    if not rows:
                        continue
                    Xk = np.stack(rows, axis=0)
                    yk = np.array([int(ins["label"]) for ins in sub_data[:len(rows)]])
                    # train/val split inside probe for fair compare
                    resk = linear_probe(Xk, yk, val_ratio=0.3, solver="saga", topk=args.topk)
                    deltas_auc.append((k, resk["auc"]))
                    deltas_acc.append((k, resk["acc"]))
            if deltas_auc:
                base_auc = deltas_auc[0][1]
                auc_curve = [(k, v - base_auc) for (k,v) in deltas_auc]
                plot_token_delta(auc_curve, ylabel="ΔAUC vs token0",
                                 title=f"{ds_tag} – Token ablation (ΔAUC)", out_path=task_out / f"{ds_tag}_token_delta_auc.png")
            if deltas_acc:
                base_acc = deltas_acc[0][1]
                acc_curve = [(k, v - base_acc) for (k,v) in deltas_acc]
                plot_token_delta(acc_curve, ylabel="ΔACC vs token0",
                                 title=f"{ds_tag} – Token ablation (ΔACC)", out_path=task_out / f"{ds_tag}_token_delta_acc.png")

        # 4) Layer curve (only when we collected hidden states)
        if H is not None and H.ndim == 3:
            # H: [N, L, D]  → per-layer linear probe
            N, L, D = H.shape
            # make labels consistent slice
            yH = y[:N]
            layer_acc = []
            for l in range(L):
                resL = linear_probe(H[:, l, :], yH, val_ratio=0.3, solver="saga", topk=min(args.topk, D))
                layer_acc.append((l, resL["acc"]))
            xs, ys = [l for l,_ in layer_acc], [v for _,v in layer_acc]
            plot_layer_curve(xs, ys, ylabel="ACC", title=f"{ds_tag} – Hidden layer curve",
                             out_path=task_out / f"{ds_tag}_layer_curve_acc.png")

        # collect results
        all_results[ds_tag] = {
            "dataset": ds_name,
            "split": split,
            "theme": theme,
            "linear_probe": lp,
        }

    # write summary
    with open(out_dir / f"{backend_name}_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] wrote {out_dir/(backend_name + '_results.json')}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qwen3b", "llava7b"], default="qwen3b",
                        help="qwen3b uses Qwen2.5-VL-3B-Instruct via HF; llava7b uses repo model builder.")
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="only used for --backend llava7b.")
    parser.add_argument("--prompt", type=str, default="oeh",
                        help="repo prompt style (see utils/prompt.py).")
    parser.add_argument("--out_dir", type=str, default="runs_task01")
    parser.add_argument("--num_samples", type=int, default=4000,
                        help="cap per dataset after chunking")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--topk", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--collect_hidden", action="store_true",
                        help="collect hidden states for layer curve (supported on qwen3b adapter)")
    parser.add_argument("--hidden_max", type=int, default=1000,
                        help="max samples for hidden-state collection")
    parser.add_argument("--token_curve", action="store_true",
                        help="build token-id Δ curves (llava7b only) for Fig.3(a,b)-style plots")
    parser.add_argument("--ablate_max", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    repo_root = Path(__file__).resolve().parent
    out_dir = ensure_dir(args.out_dir)

    if args.backend == "llava7b":
        adapter = LLaVA7BAdapter(model_path=args.model_path)
        backend_name = "llava7b"
    else:
        adapter = Qwen3BAdapter(QWEN_3B)
        backend_name = "qwen2.5-vl-3b"

    run_task(args, backend_name, adapter, repo_root, out_dir)


if __name__ == "__main__":
    main()
