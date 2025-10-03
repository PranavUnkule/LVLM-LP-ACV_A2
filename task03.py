#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3 prototypes (3.1 Beyond Classification, 3.3 Image Generation)
- (A) Phrase-grounding by first-token probing (heatmap + top box) + presentation figs
- (B) Generation reranking by first-token probing (pick best image) + presentation figs

Requires:
  pip install torch torchvision transformers qwen-vl-utils diffusers accelerate
  pip install opencv-python pillow numpy matplotlib tqdm safetensors

Example runs:
  # A) PHRASE GROUNDING / OBJECT DETECTION-ISH
  CUDA_VISIBLE_DEVICES=0,1 python task03_extensions.py detect \
    --hf_model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --images_glob "/path/to/images/*.jpg" \
    --text "Is there a dog? Answer Yes or No." \
    --grid 6 --max_images 12 --out_dir runs_t3/detect

  # B) IMAGE GENERATION RERANK
  CUDA_VISIBLE_DEVICES=0 python task03_extensions.py gen \
    --hf_model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --prompt "a red car on a snowy mountain road" \
    --num 8 --sdxl stabilityai/stable-diffusion-xl-base-1.0 \
    --out_dir runs_t3/gen
"""

from __future__ import annotations
import os, io, glob, math, json, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image as PILImage, ImageDraw
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import ImageFont

# ---------------------------
# Qwen scorer (first-token)
# ---------------------------
class QwenFirstTokenScorer:
    def __init__(self, model_id: str):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        ).eval()

        tok = self.processor.tokenizer
        self.y_ids = self._variants_to_ids(tok, ["Yes", " yes", "YES", " yes.", "Yes."])
        self.n_ids = self._variants_to_ids(tok, ["No", " no", "NO", " no.", "No."])
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def _variants_to_ids(self, tok, variants: List[str]) -> List[int]:
        ids = set()
        for v in variants:
            enc = tok(v, add_special_tokens=False).input_ids
            if len(enc) >= 1:
                ids.add(enc[0])  # first piece is good enough for Yes/No
        return sorted(ids)

    def _build_batch(self, images: List[PILImage.Image], text: str):
        messages = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}
            for img in images
        ]
        prompts = [
            self.processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        image_inputs = [[m["content"][0]["image"]] for m in messages]

        # Do NOT pass `videos` when there are none; HF will try to preprocess videos and crash.
        batch = self.processor(
            text=prompts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        for k, v in list(batch.items()):
            if hasattr(v, "to"):
                batch[k] = v.to(self.model.device)
        return batch


    @torch.inference_mode()
    def score_yes_minus_no(self, images: List[PILImage.Image], text: str) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0,), dtype=np.float32)
        batch = self._build_batch(images, text)
        out = self.model(**batch)
        input_ids = batch["input_ids"]                        # [B, T]
        B, T = input_ids.shape
        idx = (input_ids != self.pad_id).sum(dim=1) - 1       # [B] → first gen token position - 1
        rows = out.logits[torch.arange(B, device=out.logits.device), idx, :]  # [B, V]
        probs = F.softmax(rows.float(), dim=-1)               # [B, V]
        p_yes = probs[:, self.y_ids].sum(dim=-1) if self.y_ids else torch.zeros(B, device=probs.device)
        p_no  = probs[:, self.n_ids].sum(dim=-1) if self.n_ids else torch.zeros(B, device=probs.device)
        score = (p_yes - p_no).detach().cpu().numpy()         # higher => more "Yes"
        return score

# ---------------------------
# Utilities
# ---------------------------
def load_image(path: str) -> PILImage.Image:
    return PILImage.open(path).convert("RGB")

def grid_crops(img: PILImage.Image, grid: int = 6):
    W, H = img.size
    xs = np.linspace(0, W, grid+1, dtype=int).tolist()
    ys = np.linspace(0, H, grid+1, dtype=int).tolist()
    crops, boxes = [], []
    for j in range(grid):
        for i in range(grid):
            x0, x1 = int(xs[i]), int(xs[i+1])
            y0, y1 = int(ys[j]), int(ys[j+1])
            crop = img.crop((x0, y0, x1, y1))
            crops.append(crop); boxes.append((x0, y0, x1, y1))
    return crops, boxes

def compute_heat_overlay(img: PILImage.Image, boxes, scores):
    import cv2
    W, H = img.size
    heat = np.zeros((H, W), dtype=np.float32)

    # normalize scores to 0..1 (NumPy 2.0-safe)
    s = np.asarray(scores, dtype=np.float32)
    if s.size:
        rng = float(np.max(s) - np.min(s))
        if rng < 1e-6:
            s = np.zeros_like(s)
        else:
            s = (s - float(np.min(s))) / (rng + 1e-6)

    for (x0, y0, x1, y1), sc in zip(boxes, s):
        heat[y0:y1, x0:x1] = sc

    heat_rgb = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    over = PILImage.blend(img, PILImage.fromarray(heat_rgb).convert("RGB"), alpha=0.45)

    # draw top box on the original (unnormalized) scores
    if len(scores):
        k = int(np.argmax(scores))
        x0, y0, x1, y1 = boxes[k]
        drw = ImageDraw.Draw(over)
        drw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=4)
        drw.text((x0 + 6, y0 + 6), f"score={float(scores[k]):+.3f}", fill=(255, 255, 255))

    return over


def draw_top_box(overlay: PILImage.Image, boxes: List[Tuple[int,int,int,int]], scores: np.ndarray) -> PILImage.Image:
    out = overlay.copy()
    if scores.size:
        k = int(scores.argmax())
        x0,y0,x1,y1 = boxes[k]
        drw = ImageDraw.Draw(out)
        drw.rectangle([x0,y0,x1,y1], outline=(0,255,0), width=4)
        drw.text((x0+6, y0+6), f"score={scores[k]:+.3f}", fill=(255,255,255))
    return out

# ----- Presentation helpers -----
def fig_detect_panel(orig: PILImage.Image, overlay: PILImage.Image, title: str, save_path: Path):
    fig = plt.figure(figsize=(10,4), dpi=160)
    ax1 = fig.add_subplot(1,2,1); ax1.imshow(orig); ax1.set_title("Original"); ax1.axis("off")
    ax2 = fig.add_subplot(1,2,2); ax2.imshow(overlay); ax2.set_title(title); ax2.axis("off")
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)

def fig_detect_topk_crops(crops: List[PILImage.Image], scores: np.ndarray, k: int, save_path: Path):
    k = min(k, len(crops))
    order = np.argsort(-scores)[:k]
    cols = min(3, k); rows = int(math.ceil(k/cols))
    w,h = crops[0].size
    fig = plt.figure(figsize=(3*cols, 3*rows), dpi=160)
    for i, j in enumerate(order):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(crops[j]); ax.axis("off")
        ax.set_title(f"#{i+1}  {scores[j]:+.3f}", fontsize=12)
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)

def fig_hist(scores: np.ndarray, save_path: Path, title="Crop score distribution"):
    fig = plt.figure(figsize=(5,3), dpi=160)
    plt.hist(scores, bins=20, edgecolor="black")
    plt.title(title); plt.xlabel("p(Yes)-p(No)"); plt.ylabel("Count")
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)

def fig_gen_scores_bar(scores: np.ndarray, save_path: Path, prompt: str):
    order = np.argsort(-scores)
    fig = plt.figure(figsize=(8,3), dpi=160)
    plt.bar(np.arange(len(scores)), scores[order])
    plt.title("Candidate images ranked by first-token score")
    plt.xlabel("Rank (best → worst)"); plt.ylabel("p(Yes)-p(No)")
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)

def fig_pipeline(save_path: Path, title: str, boxes: list[str], arrows: list[tuple[int,int]]):
    fig = plt.figure(figsize=(8,2.2), dpi=160)
    ax = fig.add_subplot(111); ax.axis("off")
    # place boxes in a row
    x0, y, dx, dy = 0.05, 0.5, 0.22, 0.35
    centers = []
    for i, text in enumerate(boxes):
        x = x0 + i*(dx+0.05)
        rect = FancyBboxPatch((x, y-dy/2), dx, dy, boxstyle="round,pad=0.02", fc="#e7f0ff", ec="#1f77b4", lw=1.5)
        ax.add_patch(rect)
        ax.text(x+dx/2, y, text, ha="center", va="center", fontsize=10)
        centers.append((x+dx, y))
    for a,b in arrows:
        xA,yA = centers[a]
        xB,yB = centers[b][0]-dx, centers[b][1]
        ax.annotate("", xy=(xB,yB), xytext=(xA+0.02,yA),
                    arrowprops=dict(arrowstyle=ArrowStyle("->", head_length=8, head_width=4), lw=1.5))
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    plt.title(title)
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)

# ---------------------------
# A) DETECT / GROUNDING
# ---------------------------
def run_detect(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scorer = QwenFirstTokenScorer(args.hf_model_id)

    paths = sorted(glob.glob(args.images_glob))[: args.max_images or None]
    assert paths, f"No images matched: {args.images_glob}"

    # tiny pipeline diagram for slides
    fig_pipeline(out/"pipeline_detect.png",
                 title="Phrase Grounding via First-Token Probing",
                 boxes=["Image", "Grid Crops", "Qwen first token\np(Yes)-p(No)", "Heatmap + Top Box"],
                 arrows=[(0,1),(1,2),(2,3)])

    summary = []
    pbar = tqdm(paths, desc="detect (phrase grounding)")
    for p in pbar:
        img = load_image(p)
        crops, boxes = grid_crops(img, grid=args.grid)
        scores = scorer.score_yes_minus_no(crops, args.text)

        overlay = compute_heat_overlay(img, boxes, scores)
        overlay_box = draw_top_box(overlay, boxes, scores)

        stem = Path(p).stem
        # base artifacts
        overlay_box.save(out / f"{stem}_heatmap.png")

        # presentation figs
        fig_detect_panel(img, overlay_box,
                         f"Heatmap (grid={args.grid}) — top score {scores.max():+.3f}",
                         out / f"{stem}_panel.png")
        fig_detect_topk_crops(crops, scores, k=9, save_path=out / f"{stem}_topk_crops.png")
        fig_hist(scores, out / f"{stem}_scores_hist.png")

        best_box = None
        if scores.size:
            k = int(scores.argmax()); best_box = boxes[k]
        summary.append({
            "image": p, "best_score": float(scores.max() if len(scores) else 0.0),
            "best_box": best_box, "text": args.text,
            "grid": args.grid, "n_cells": len(boxes)
        })

    with open(out / "detect_summary.json", "w") as f:
        json.dump(_py(summary), f, indent=2)
    print(f"[detect] wrote {out/'detect_summary.json'}")
    print(f"[detect] presentation assets: pipeline_detect.png, *_panel.png, *_topk_crops.png, *_scores_hist.png")

# ---------------------------
# B) GENERATION RERANK
# ---------------------------

def _py(o):
    import numpy as np
    if isinstance(o, dict):  return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_py(x) for x in o]
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    return o

def run_gen(args):
    import gc
    from diffusers import StableDiffusionXLPipeline
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # --- A) GENERATE FIRST (SDXL on GPU), then free it ---------------------------------
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.sdxl,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to("cuda")

    # memory helpers
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    prompt = args.prompt.strip()
    neg = args.negative_prompt or ""
    g = torch.Generator(device="cuda").manual_seed(args.seed)

    imgs: List[PILImage.Image] = []
    for _ in tqdm(range(args.num), desc="generate"):
        img = pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            height=args.height,
            width=args.width,
            generator=g,
        ).images[0]
        imgs.append(img)

    # free SDXL before loading Qwen
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    # --- B) SCORE AFTER (Qwen on GPU, SDXL is gone) ------------------------------------
    scorer = QwenFirstTokenScorer(args.hf_model_id)

    judge_text = f"Does this image match the text: '{prompt}'? Answer Yes or No."
    scores = []
    bs = 8
    for i in range(0, len(imgs), bs):
        scores.extend(scorer.score_yes_minus_no(imgs[i:i+bs], judge_text).tolist())
    scores = np.array(scores, dtype=np.float32)

    # pick best, save grid
    order = np.argsort(-scores)
    best = imgs[int(order[0])]
    best.save(out / "best.png")

    FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    font = None


    cols = min(4, args.num); rows = int(math.ceil(args.num/cols))
    w, h = imgs[0].size
    for fp in FONT_CANDIDATES:
        try:
            # Try a very visible size; tweak the divisor (5 → larger, 6 → smaller)
            font_size = max(72, w // 12)
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()
        font_size = max(72, w // 12)
    sheet = PILImage.new("RGB", (cols*w, rows*h), (0,0,0))
    for idx, j in enumerate(order):
        r, c = divmod(idx, cols)
        score_text = f"{scores[j]:+.3f}"
        tile = imgs[j].copy()
        draw = ImageDraw.Draw(tile)
        bbox = draw.textbbox((0, 0), score_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = max(24, font_size // 6)
        x0, y0 = 8, 8
        x1, y1 = x0 + tw + 2*pad, y0 + th + 2*pad
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
        draw.text(
            (x0 + pad, y0 + pad),
            score_text,
            font=font,
            fill=(0, 0, 0),
            stroke_width=max(2, font_size // 14),
            stroke_fill=(255, 255, 255),
        )
        sheet.paste(tile, (c*w, r*h))
    sheet.save(out / "grid_scored.png")

    meta = {
        "prompt": prompt,
        "negative_prompt": neg,
        "scores": scores.tolist(),
        "order": order.tolist(),
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "scale": args.scale,
        "seed": args.seed,
    }
    with open(out / "gen_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[gen] wrote {out/'best.png'} and {out/'grid_scored.png'}")


# ---------------------------
# CLI
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("detect", help="Phrase grounding by first-token probing")
    pd.add_argument("--hf_model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    pd.add_argument("--images_glob", type=str, required=True)
    pd.add_argument("--text", type=str, required=True,
                    help="Ask as Yes/No, e.g., 'Is there a dog? Answer Yes or No.'")
    pd.add_argument("--grid", type=int, default=6)
    pd.add_argument("--max_images", type=int, default=12)
    pd.add_argument("--out_dir", type=str, default="runs_t3/detect")
    pd.set_defaults(func=run_detect)

    pg = sub.add_parser("gen", help="Generation reranking by first-token probing")
    pg.add_argument("--hf_model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    pg.add_argument("--prompt", type=str, required=True)
    pg.add_argument("--negative_prompt", type=str, default="")
    pg.add_argument("--sdxl", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    pg.add_argument("--num", type=int, default=8)
    pg.add_argument("--steps", type=int, default=30)
    pg.add_argument("--scale", type=float, default=7.0)
    pg.add_argument("--height", type=int, default=768)
    pg.add_argument("--width",  type=int, default=768)
    pg.add_argument("--seed", type=int, default=42)
    pg.add_argument("--out_dir", type=str, default="runs_t3/gen")
    pg.set_defaults(func=run_gen)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
