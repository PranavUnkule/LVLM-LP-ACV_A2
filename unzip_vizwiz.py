#!/usr/bin/env python3
import argparse, zipfile, shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def safe_extract_subset(zip_path: Path, dest_dir: Path, only_ext=IMG_EXTS) -> list[Path]:
    """Extract zip into dest_dir (no traversal), return list of extracted files (absolute)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if only_ext and name.suffix.lower() not in {e.lower() for e in only_ext}:
                continue
            target = dest_dir / name.name  # flatten to filename only
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out

def move_all(files: list[Path], split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = split_dir / f.name
        if dest.exists() and dest.resolve() != f.resolve():
            stem, ext = f.stem, f.suffix
            i = 1
            while (split_dir / f"{stem}_{i}{ext}").exists():
                i += 1
            dest = split_dir / f"{stem}_{i}{ext}"
        shutil.move(str(f), str(dest))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--train-zip", required=True)
    ap.add_argument("--val-zip", required=True)
    ap.add_argument("--anno-zip", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    base = repo / "data" / "VizWiz"
    img_root = base / "images"
    ann_root = base / "annotations"
    train_dir = img_root / "train"
    val_dir   = img_root / "val"

    if base.exists() and args.force:
        shutil.rmtree(base)
    img_root.mkdir(parents=True, exist_ok=True)
    ann_root.mkdir(parents=True, exist_ok=True)

    # 1) Train images -> extract into tmp and then move into images/train
    tmp_train = img_root / "_tmp_train"
    files_train = safe_extract_subset(Path(args.train_zip), tmp_train)
    move_all(files_train, train_dir)
    shutil.rmtree(tmp_train, ignore_errors=True)

    # 2) Val images -> tmp then move into images/val
    tmp_val = img_root / "_tmp_val"
    files_val = safe_extract_subset(Path(args.val_zip), tmp_val)
    move_all(files_val, val_dir)
    shutil.rmtree(tmp_val, ignore_errors=True)

    # 3) Annotations: extract jsons (train.json, val.json) to annotations/
    with zipfile.ZipFile(args.anno_zip) as z:
        for info in z.infolist():
            if info.is_dir(): 
                continue
            name = Path(info.filename).name.lower()
            if name in {"train.json", "val.json"}:
                with z.open(info) as src, open(ann_root / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    print("[done]")
    print(f" Train images: {len(list(train_dir.glob('*.jpg')))} -> {train_dir}")
    print(f" Val images:   {len(list(val_dir.glob('*.jpg')))} -> {val_dir}")
    print(f" Annotations:  {list(ann_root.glob('*.json'))} -> {ann_root}")

if __name__ == "__main__":
    main()
