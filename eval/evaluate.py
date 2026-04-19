import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold


def load_bin(path):
    with open(path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")

    images = []
    for b in bins:
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        images.append(np.transpose(img, (2, 0, 1)))
    images = np.stack(images).astype(np.float32)
    images = (images - 127.5) / 127.5

    return images, np.array(issame, dtype=bool)


@torch.no_grad()
def extract_embeddings(model, images, batch_size=256, device="cuda"):
    model.eval()
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[i:i + batch_size]).to(device)
        emb = F.normalize(model(batch), dim=1)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings)


def _pair_scores(embeddings):
    return np.sum(embeddings[0::2] * embeddings[1::2], axis=1)


def compute_pair_scores(embeddings, issame):
    scores = _pair_scores(embeddings)
    return {
        "scores": scores.tolist(),
        "issame": issame.astype(bool).tolist(),
    }


def compute_accuracy(embeddings, issame, nfolds=10, threshold_step=0.005):
    scores = _pair_scores(embeddings)
    thresholds = np.arange(-1, 1, threshold_step)
    kfold = KFold(n_splits=nfolds, shuffle=False)

    accuracies = []
    best_thresholds = []
    for train_idx, test_idx in kfold.split(scores):
        train_preds = scores[train_idx][None, :] > thresholds[:, None]
        train_acc = (train_preds == issame[train_idx][None, :]).mean(axis=1)
        best_th = thresholds[train_acc.argmax()]

        test_acc = np.mean((scores[test_idx] > best_th) == issame[test_idx])
        accuracies.append(test_acc)
        best_thresholds.append(best_th)

    return float(np.mean(accuracies)), float(np.std(accuracies)), best_thresholds


def compute_tar_at_far(embeddings, issame, far_targets=(1e-3, 1e-4)):
    scores = _pair_scores(embeddings)
    fpr, tpr, _ = roc_curve(issame, scores)
    return {f"TAR@FAR={far}": float(np.interp(far, fpr, tpr)) for far in far_targets}


def compute_roc_stats(embeddings, issame):
    scores = _pair_scores(embeddings)
    fpr, tpr, thresholds = roc_curve(issame, scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    auc = float(roc_auc_score(issame, scores))
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "eer": eer,
        "auc": auc,
    }


def evaluate_benchmark(model, bin_path, device="cuda", batch_size=256):
    images, issame = load_bin(bin_path)
    embeddings = extract_embeddings(model, images, batch_size, device)
    accuracy, std, _ = compute_accuracy(embeddings, issame)
    pair_stats = compute_pair_scores(embeddings, issame)
    roc_stats = compute_roc_stats(embeddings, issame)
    return {"accuracy": accuracy, "std": std, **compute_tar_at_far(embeddings, issame), **pair_stats, **roc_stats}


def evaluate_all(model, eval_root, benchmarks=None, device="cuda", batch_size=256):
    if benchmarks is None:
        benchmarks = ["lfw", "cfp_fp", "agedb_30"]

    results = {}
    for name in benchmarks:
        bin_path = os.path.join(eval_root, f"{name}.bin")
        if not os.path.exists(bin_path):
            print(f"  {name}: SKIPPED (file not found)")
            continue
        results[name] = evaluate_benchmark(model, bin_path, device, batch_size)
        print(f"  {name}: {results[name]['accuracy']:.4f} ± {results[name]['std']:.4f}")

    if results:
        avg = float(np.mean([r["accuracy"] for r in results.values()]))
        print(f"  average: {avg:.4f}")

    return results
