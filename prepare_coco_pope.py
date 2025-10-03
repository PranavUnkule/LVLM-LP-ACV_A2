#!/usr/bin/env python3
"""
Unpack COCO 2014 images for POPE into the LVLM-LP expected layout.

Creates:
  <repo_root>/data/COCO/images/train2014/*.jpg
  <repo_root>/data/COCO/images/val2014/*.jpg

Usage:
  python prepare_coco_pope.py \
    --repo-root /path/to/LVLM-LP \
    --train2014-zip /path/to/train2014.zip \
    --val2014-zip   /path/to/val2014.zip \
    [--force]
"""

import argparse
import zipfile
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def safe_extract_images(zip_path: Path, dest_dir: Path) -> list[Path]:
    """
    Extract only image files from zip to a temporary directory (flattened),
    guarding against zip-slip. Returns absolute paths of extracted files.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            inner = Path(info.filename)
            if inner.suffix.lower() not in IMG_EXTS:
                continue
            # flatten to just the file name (COCO zips have train2014/ or val2014/ prefixes)
            target = dest_dir / inner.name
            # write file
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out


def move_all(files: list[Path], split_dir: Path) -> int:
    """
    Move the extracted files into the final split directory.
    Handles name collisions by appending _1, _2, ...
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in files:
        dest = split_dir / f.name
        if dest.exists() and dest.resolve() != f.resolve():
            stem, ext = f.stem, f.suffix
            i = 1
            while (split_dir / f"{stem}_{i}{ext}").exists():
                i += 1
            dest = split_dir / f"{stem}_{i}{ext}"
        shutil.move(str(f), str(dest))
        moved += 1
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True, help="Path to LVLM-LP repo root")
    ap.add_argument("--train2014-zip", required=True, help="Path to COCO train2014.zip")
    ap.add_argument("--val2014-zip",   required=True, help="Path to COCO val2014.zip")
    ap.add_argument("--force", action="store_true", help="Delete existing data/COCO before extracting")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    base = repo / "data" / "COCO"
    img_root = base / "images"
    train_dir = img_root / "train2014"
    val_dir   = img_root / "val2014"

    # Clean if requested
    if base.exists() and args.force:
        shutil.rmtree(base)
    img_root.mkdir(parents=True, exist_ok=True)

    # 1) Train
    tmp_train = img_root / "_tmp_train2014"
    files_train = safe_extract_images(Path(args.train2014_zip), tmp_train)
    n_train = move_all(files_train, train_dir)
    shutil.rmtree(tmp_train, ignore_errors=True)

    # 2) Val
    tmp_val = img_root / "_tmp_val2014"
    files_val = safe_extract_images(Path(args.val2014_zip), tmp_val)
    n_val = move_all(files_val, val_dir)
    shutil.rmtree(tmp_val, ignore_errors=True)

    print("[done]")
    print(f" Train2014 images: {n_train} -> {train_dir}")
    print(f" Val2014 images:   {n_val} -> {val_dir}")
    # Typical counts: train2014 ≈ 82,783; val2014 ≈ 40,504 (for full COCO 2014)


if __name__ == "__main__":
    main()
