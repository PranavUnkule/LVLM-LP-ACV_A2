#!/usr/bin/env python3
"""
Create LVLM-LP-compatible MathVista correctness labels with Qwen2.5-VL-7B.

Writes:
  ./output/<split>/MathV_output.json
Saves images under:
  ./data/MathVista/images/<split>/*.jpg

Usage (recommended split from the paper: testmini):
  python make_mathvista_output.py --repo-root /workspace/LVLM-LP --split testmini
"""

import os, re, json, argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image as PILImage
from datasets import load_dataset
import torch

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # switched to 7B

from PIL import Image as PILImage
import os

def _get_pil_from_example(ex):
    """
    Prefer HF's decoded PIL image when available.
    Fallback to 'image' (path/bytes) if needed.
    """
    img = ex.get("decoded_image", None)
    if isinstance(img, PILImage.Image):
        return img.convert("RGB")

    # Fallbacks
    img = ex.get("image", None)
    if isinstance(img, PILImage.Image):
        return img.convert("RGB")
    if isinstance(img, str) and os.path.exists(img):
        try:
            return PILImage.open(img).convert("RGB")
        except Exception:
            return None
    return None

def _get_question_from_example(ex):
    """
    Prefer 'question'; fall back to 'query'.
    """
    q = ex.get("question", None)
    if q and isinstance(q, str) and q.strip():
        return q.strip()
    q = ex.get("query", None)
    if q and isinstance(q, str) and q.strip():
        return q.strip()
    return None


def normalize_ans(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,\.;:!?\(\)\[\]\{\}\"'`]", "", s)
    return s


def build_inputs(processor, image: PILImage.Image, question: str, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": f"Answer the question concisely. {question}"},
        ],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(device)
    return inputs


@torch.inference_mode()
def generate_pred(processor, model, image: PILImage.Image, question: str, max_new_tokens=48) -> str:
    device = next(model.parameters()).device
    inputs = build_inputs(processor, image, question, device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return text.strip()


def save_image(img: PILImage.Image, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{name}.jpg"
    img.convert("RGB").save(fp, quality=92)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="LVLM-LP repo root (where ./output will be created)")
    ap.add_argument("--split", default="testmini", choices=["testmini", "val", "train"], help="MathVista split")
    ap.add_argument("--limit", type=int, default=None, help="cap #examples for a quick run")
    ap.add_argument("--hf_model_id", type=str, default=DEFAULT_MODEL_ID, help="HF model id to use")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    repo_root = Path(args.repo_root).resolve()
    out_json = repo_root / "output" / args.split / "MathV_output.json"
    out_img_dir = repo_root / "data" / "MathVista" / "images" / args.split

    # 1) Model (7B by default; L40s → bfloat16 is ideal)
    print(f"[init] loading {args.hf_model_id} ...")
    processor = AutoProcessor.from_pretrained(args.hf_model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.hf_model_id,
        dtype=dtype,
        device_map="auto",
    ).eval()

    # 2) Dataset (HF). The repo note says testmini is auto-downloaded; we mirror that.
    print(f"[data] loading MathVista split={args.split} from Hugging Face...")
    try:
        ds = load_dataset("AI4Math/MathVista", split=args.split)
    except Exception:
        ds = load_dataset("lukaemon/MathVista", split=args.split)  # fallback mirror

    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    rows = []
    skipped = 0

    for i, ex in enumerate(ds):
        img = _get_pil_from_example(ex)
        q   = _get_question_from_example(ex)
        gt  = str(ex.get("answer", "")).strip()

        if img is None or q is None or not gt:
            skipped += 1
            continue

        # save image locally for LVLM-LP dataset class
        img_path = save_image(img, out_img_dir, f"mv_{args.split}_{i:06d}")

        # predict with Qwen 7B
        pred = generate_pred(processor, model, img, q)

        # EM-style label (normalize both sides)
        label = 1 if normalize_ans(pred) == normalize_ans(gt) else 0

        rows.append({
            "img_path": str(img_path),
            "question": q,
            "gt_answer": gt,
            "pred_answer": pred,
            "label": label,  # 1 = correct, 0 = incorrect
        })

        if (i + 1) % 25 == 0:
            print(f"[progress] {i+1}/{len(ds)}  kept={len(rows)}  skipped={skipped}")


    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rows, f, ensure_ascii=False)

    print(f"[done] wrote {out_json} with {len(rows)} items")
    print(f"      images under {out_img_dir}")


if __name__ == "__main__":
    main()
