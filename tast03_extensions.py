#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3 prototypes (3.1 Beyond Classification, 3.3 Image Generation)
- (A) Phrase-grounding by first-token probing (heatmap + top box)
- (B) Generation reranking by first-token probing (pick best image)

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
from PIL import Image as PILImage, ImageDraw, ImageFont
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ---------------------------
# Qwen scorer (first-token)
# ---------------------------
class QwenFirstTokenScorer:
    def __init__(self, model_id: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        ).eval()

        tok = self.processor.tokenizer
        # Collect a few token-id variants for "Yes"/"No"
        self.y_ids = self._variants_to_ids(tok, ["Yes", " yes", "YES", " yes.", "Yes."])
        self.n_ids = self._variants_to_ids(tok, ["No", " no", "NO", " no.", "No."])
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def _variants_to_ids(self, tok, variants: List[str]) -> List[int]:
        ids = set()
        for v in variants:
            enc = tok(v, add_special_tokens=False).input_ids
            if len(enc) >= 1:
                ids.add(enc[0])  # take first piece as proxy; good enough for Yes/No
        return sorted(ids)

    def _build_batch(self, images: List[PILImage.Image], text: str):
        # build batched chat prompts and inputs
        messages = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}
            for img in images
        ]
        prompts = [
            self.processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        # processor supports list-of-lists for images/videos
        image_inputs = [[m["content"][0]["image"]] for m in messages]
        video_inputs = [[] for _ in messages]
        batch = self.processor(text=prompts, images=image_inputs, videos=video_inputs,
                               return_tensors="pt", padding=True)
        for k, v in list(batch.items()):
            if hasattr(v, "to"):
                batch[k] = v.to(self.model.device)
        return batch

    @torch.inference_mode()
    def score_yes_minus_no(self, images: List[PILImage.Image], text: str) -> np.ndarray:
        """
        Returns a float score per image:  sum_p(Yes) - sum_p(No) at the first generated token.
        """
        if len(images) == 0:
            return np.zeros((0,), dtype=np.float32)

        batch = self._build_batch(images, text)
        out = self.model(**batch)  # forward only
        input_ids = batch["input_ids"]                        # [B, T]
        B, T = input_ids.shape
        # last input index per sample = first gen token position - 1
        idx = (input_ids != self.pad_id).sum(dim=1) - 1       # [B]
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
    img = PILImage.open(path).convert("RGB")
    return img

def grid_crops(img: PILImage.Image, grid: int = 6) -> Tuple[List[PILImage.Image], List[Tuple[int,int,int,int]]]:
    W, H = img.size
    xs = np.linspace(0, W, grid+1, dtype=int)
    ys = np.linspace(0, H, grid+1, dtype=int)
    crops, boxes = [], []
    for j in range(grid):
        for i in range(grid):
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[j], ys[j+1]
            crop = img.crop((x0, y0, x1, y1))
            crops.append(crop); boxes.append((x0, y0, x1, y1))
    return crops, boxes

def draw_heatmap_and_box(img: PILImage.Image, boxes: List[Tuple[int,int,int,int]], scores: np.ndarray,
                         out_path: Path, title: str = None):
    import cv2
    W, H = img.size
    heat = np.zeros((H, W), dtype=np.float32)
    # normalize scores to 0..1
    s = scores.copy()
    if s.size:
        s = (s - s.min()) / (s.ptp() + 1e-6)
    # fill each cell with its score
    n = int(math.sqrt(len(boxes)))
    for (x0,y0,x1,y1), sc in zip(boxes, s):
        heat[y0:y1, x0:x1] = sc
    heat_rgb = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    over = PILImage.blend(img, PILImage.fromarray(heat_rgb).convert("RGB"), alpha=0.45)

    # top box
    if scores.size:
        k = int(scores.argmax())
        x0,y0,x1,y1 = boxes[k]
        drw = ImageDraw.Draw(over)
        drw.rectangle([x0,y0,x1,y1], outline=(0,255,0), width=4)
        txt = f"score={scores[k]:+.3f}"
        drw.text((x0+6, y0+6), txt, fill=(255,255,255))

    over.save(out_path)

# ---------------------------
# A) DETECT / GROUNDING
# ---------------------------
def run_detect(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scorer = QwenFirstTokenScorer(args.hf_model_id)

    paths = sorted(glob.glob(args.images_glob))[: args.max_images or None]
    assert paths, f"No images matched: {args.images_glob}"

    summary = []
    pbar = tqdm(paths, desc="detect (phrase grounding)")
    for p in pbar:
        img = load_image(p)
        crops, boxes = grid_crops(img, grid=args.grid)
        # batch score all crops
        scores = scorer.score_yes_minus_no(crops, args.text)
        # save overlay
        out_img = out / (Path(p).stem + "_heatmap.png")
        draw_heatmap_and_box(img, boxes, scores, out_img)
        summary.append({"image": p, "best_score": float(scores.max() if len(scores) else 0.0),
                        "best_box": boxes[int(scores.argmax())] if len(scores) else None,
                        "text": args.text, "grid": args.grid, "n_cells": len(boxes)})
    with open(out / "detect_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[detect] wrote {out/'detect_summary.json'}")

# ---------------------------
# B) GENERATION RERANK
# ---------------------------
def run_gen(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scorer = QwenFirstTokenScorer(args.hf_model_id)

    # Load SDXL
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.sdxl, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    prompt = args.prompt.strip()
    neg = args.negative_prompt or ""
    g = torch.Generator(device="cuda").manual_seed(args.seed)

    imgs: List[PILImage.Image] = []
    for i in tqdm(range(args.num), desc="generate"):
        img = pipe(prompt=prompt, negative_prompt=neg, num_inference_steps=args.steps,
                   guidance_scale=args.scale, generator=g).images[0]
        imgs.append(img)

    # score with first token Yes/No
    judge_text = f"Does this image match the text: '{prompt}'? Answer Yes or No."
    scores = []
    # score in small batches to keep memory low
    bs = 8
    for i in range(0, len(imgs), bs):
        scores.extend(scorer.score_yes_minus_no(imgs[i:i+bs], judge_text).tolist())
    scores = np.array(scores, dtype=np.float32)

    # pick best, save grid
    order = np.argsort(-scores)
    best = imgs[int(order[0])]
    best.save(out / "best.png")
    # grid sheet
    cols = min(4, args.num); rows = int(math.ceil(args.num/cols))
    w,h = imgs[0].size
    sheet = PILImage.new("RGB", (cols*w, rows*h), (0,0,0))
    for idx, j in enumerate(order):
        r, c = divmod(idx, cols)
        tile = imgs[j].copy()
        draw = ImageDraw.Draw(tile)
        draw.rectangle([5,5, w-5, 60], fill=(0,0,0))
        draw.text((10,10), f"{scores[j]:+.3f}", fill=(255,255,255))
        sheet.paste(tile, (c*w, r*h))
    sheet.save(out / "grid_scored.png")

    meta = {"prompt": prompt, "negative_prompt": neg, "scores": scores.tolist(), "order": order.tolist()}
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
    pg.add_argument("--seed", type=int, default=42)
    pg.add_argument("--out_dir", type=str, default="runs_t3/gen")
    pg.set_defaults(func=run_gen)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()