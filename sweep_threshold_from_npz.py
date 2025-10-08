#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

def topk_by_variance(X, k):
    if k is None or k >= X.shape[1]: return X, None
    var = X.var(axis=0)
    idx = np.argsort(var)[-k:]
    return X[:, idx], idx

def fit_probe(Xtr, ytr, max_iter=500, n_jobs=64):
    clf = LogisticRegression(max_iter=max_iter, solver="saga", n_jobs=n_jobs)
    clf.fit(Xtr, ytr)
    W = torch.from_numpy(clf.coef_.astype(np.float32))      # [1, K]
    b = torch.from_numpy(clf.intercept_.astype(np.float32)) # [1]
    return W, b

def scores_from_linear(X, W, b):
    # X: [N,K] or [N,V];  W:[1,K]; b:[1]
    z = torch.from_numpy(X).float() @ W.T + b  # [N,1]
    return torch.sigmoid(z).squeeze(1).numpy() # [N]

def bin_stats(y_true, pred_pos):
    tn = np.sum((y_true==0) & (pred_pos==0))
    fp = np.sum((y_true==0) & (pred_pos==1))
    fn = np.sum((y_true==1) & (pred_pos==0))
    tp = np.sum((y_true==1) & (pred_pos==1))
    tpr = tp / (tp+fn+1e-9)
    fpr = fp / (fp+tn+1e-9)
    prec = tp / (tp+fp+1e-9)
    rec  = tpr
    f1   = 2*prec*rec / (prec+rec+1e-9)
    return dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), tpr=tpr, fpr=fpr, f1=f1)

def pick_thresholds(y_val, p_val, target_fa=0.05, safe_is_one=True):
    # Accept if p >= t  (since p=prob(safe))
    thr_candidates = np.unique(p_val)
    best_f1 = (-1, 0.5)
    best_j  = (-1, 0.5)

    # For false-accept: FA = accept rate on UNSAFE class
    unsafe_label = 0 if safe_is_one else 1

    best_fa_thr = None
    best_fa_acc = None
    for t in thr_candidates:
        pred = (p_val >= t).astype(int)            # accept/positive
        stats = bin_stats(y_val, pred)
        J = stats["tpr"] - stats["fpr"]
        if stats["f1"] > best_f1[0]:
            best_f1 = (stats["f1"], t)
        if J > best_j[0]:
            best_j = (J, t)
        # false-accept on unsafe:
        mask_unsafe = (y_val == unsafe_label)
        fa = np.mean((p_val[mask_unsafe] >= t)) if np.any(mask_unsafe) else 0.0
        # choose the **largest** threshold that keeps FA <= target (more conservative)
        if fa <= target_fa:
            if best_fa_thr is None or t > best_fa_thr:
                best_fa_thr, best_fa_acc = t, fa

    return {
        "f1_opt": {"f1": float(best_f1[0]), "threshold": float(best_f1[1])},
        "youden_j_opt": {"J": float(best_j[0]), "threshold": float(best_j[1])},
        "target_FA": {"FA_target": float(target_fa),
                      "threshold": None if best_fa_thr is None else float(best_fa_thr),
                      "FA_at_threshold": None if best_fa_acc is None else float(best_fa_acc)}
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Task01 logits NPZ (keys: X, y)")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=4096)
    ap.add_argument("--max_iter", type=int, default=500)
    ap.add_argument("--n_jobs", type=int, default=64)
    ap.add_argument("--target_fa", type=float, default=0.05,
                    help="Max false-accept rate on UNSAFE class")
    ap.add_argument("--safe_is_one", action="store_true", default=True,
                    help="If labels use 1=safe, 0=unsafe (default True). Use --no-safe_is_one to flip.")
    ap.add_argument("--no-safe_is_one", dest="safe_is_one", action="store_false")
    ap.add_argument("--limit", type=int, default=None,
                help="Use only the first N samples from the NPZ")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_json", type=str, default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)
    z = np.load(args.npz)
    X, y = z["X"], z["y"].astype(int)

    if args.limit is not None:
        X = X[:args.limit]
        y = y[:args.limit]

    n = len(y)
    n_val = max(1, int(np.ceil(n * args.val_ratio)))
    idx = np.arange(n)
    # simple deterministic split: last chunk = val
    tr_idx, val_idx = idx[:-n_val], idx[-n_val:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xval, yval = X[val_idx], y[val_idx]

    Xtr_k, keep = topk_by_variance(Xtr, args.topk)
    W, b = fit_probe(Xtr_k, ytr, max_iter=args.max_iter, n_jobs=args.n_jobs)

    if keep is not None:
        Xval_k = Xval[:, keep]
    else:
        Xval_k = Xval

    p_val = scores_from_linear(Xval_k, W, b)

    results = pick_thresholds(yval, p_val, target_fa=args.target_fa, safe_is_one=args.safe_is_one)

    summary = {
        "npz": args.npz,
        "n_total": int(n),
        "n_train": int(len(ytr)),
        "n_val": int(len(yval)),
        "pos_rate_in_val": float(np.mean(yval==1)),
        "safe_is_one": bool(args.safe_is_one),
        "topk": None if args.topk is None else int(args.topk),
        "probe": {"max_iter": int(args.max_iter), "n_jobs": int(args.n_jobs)},
        "thresholds": results
    }
    print(json.dumps(summary, indent=2))
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
