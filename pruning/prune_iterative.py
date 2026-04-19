import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_dataloader
from eval.evaluate import evaluate_all
from losses.arcface import ArcFace
from models import get_model
from train.optimizer import build_optimizer
from train.train import load_config, train_one_epoch


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model):
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_total += sum(buf.numel() * buf.element_size() for buf in model.buffers())
    return bytes_total


def write_csv(history, benchmarks, csv_path):
    fieldnames = [
        "iteration", "cumulative_ratio", "params", "params_fraction",
        "size_mb", "avg_acc", "acc_drop", *benchmarks, "time_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for entry in history:
            row = {
                "iteration": entry["iteration"],
                "cumulative_ratio": round(entry["cumulative_ratio"], 4),
                "params": entry["params"],
                "params_fraction": round(entry["params_fraction"], 4),
                "size_mb": round(entry["size_bytes"] / 1024 / 1024, 3),
                "avg_acc": round(entry["avg_acc"], 4),
                "acc_drop": round(entry["acc_drop"], 4),
                "time_s": round(entry.get("time_s", 0), 1),
            }
            for bm in benchmarks:
                row[bm] = round(entry["per_benchmark"].get(bm, 0), 4)
            w.writerow(row)


def write_eval_csv(stage, eval_results, csv_path):
    fieldnames = ["stage", "benchmark", "accuracy", "std", "eer", "auc", "tar_far_1e-3", "tar_far_1e-4"]
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for benchmark, metrics in eval_results.items():
            w.writerow({
                "stage": stage,
                "benchmark": benchmark,
                "accuracy": round(metrics.get("accuracy", 0), 6),
                "std": round(metrics.get("std", 0), 6),
                "eer": round(metrics.get("eer", 0), 6),
                "auc": round(metrics.get("auc", 0), 6),
                "tar_far_1e-3": round(metrics.get("TAR@FAR=0.001", 0), 6),
                "tar_far_1e-4": round(metrics.get("TAR@FAR=0.0001", 0), 6),
            })


def write_roc_json(stage, eval_results, json_path):
    existing = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
    existing[stage] = {
        benchmark: {
            "fpr": metrics.get("fpr", []),
            "tpr": metrics.get("tpr", []),
            "thresholds": metrics.get("thresholds", []),
            "eer": metrics.get("eer", 0),
            "auc": metrics.get("auc", 0),
        }
        for benchmark, metrics in eval_results.items()
    }
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)


def write_scores_json(stage, eval_results, json_path):
    existing = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
    existing[stage] = {
        benchmark: {
            "scores": metrics.get("scores", []),
            "issame": metrics.get("issame", []),
        }
        for benchmark, metrics in eval_results.items()
    }
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)


def get_ignored_layers(model, embedding_dim):
    ignored = []
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            ignored.append(m)
        elif isinstance(m, nn.Linear) and m.out_features == embedding_dim:
            ignored.append(m)
        elif isinstance(m, nn.Conv2d) and m.out_channels == embedding_dim:
            ignored.append(m)
    return ignored


def eval_model(model, config, device):
    eval_root = config["eval"]["eval_root"]
    benchmarks = config["eval"].get("benchmarks", ["lfw", "cfp_fp", "agedb_30"])
    results = evaluate_all(model, eval_root, benchmarks, device=str(device))
    if not results:
        return 0.0, {}
    avg = sum(r["accuracy"] for r in results.values()) / len(results)
    return avg, {k: v["accuracy"] for k, v in results.items()}


def recalibrate_bn(model, dataloader, device, num_batches):
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            _ = model(images.to(device, non_blocking=True))


def train_one_epoch_kd(student, teacher, head, dataloader, optimizer, scaler, device, kd_weight, log_interval):
    student.train()
    teacher.eval()
    head.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="KD-Training")
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            student_emb = student(images)
            logits = head(student_emb, labels)
            arc_loss = criterion(logits, labels)

            with torch.no_grad():
                teacher_emb = teacher(images)

            distill_loss = 1 - F.cosine_similarity(student_emb.float(), teacher_emb.float(), dim=1).mean()
            loss = arc_loss + kd_weight * distill_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += images.size(0)

        if (i + 1) % log_interval == 0:
            pbar.set_postfix(arc=f"{arc_loss.item():.4f}", kd=f"{distill_loss.item():.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step_ratio", type=float, default=0.1)
    parser.add_argument("--finetune_epochs", type=int, default=2)
    parser.add_argument("--max_acc_drop", type=float, default=0.02)
    parser.add_argument("--min_params_ratio", type=float, default=0.01)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--final_finetune_epochs", type=int, default=3)
    parser.add_argument("--use_kd", action="store_true")
    parser.add_argument("--kd_weight", type=float, default=1.0)
    parser.add_argument("--bn_recal_batches", type=int, default=100)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--finetune_lr", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.data_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["train"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["train"]["seed"])

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    dataloader, num_classes = build_dataloader(config)

    model = get_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    head = ArcFace(
        embedding_dim=config["model"]["embedding_dim"],
        num_classes=num_classes,
        margin=config["loss"]["margin"],
        scale=config["loss"]["scale"],
    ).to(device)
    if "head" in ckpt:
        head.load_state_dict(ckpt["head"])

    teacher = None
    if args.use_kd:
        print("Loading teacher for KD...")
        teacher = get_model(config).to(device)
        teacher.load_state_dict(ckpt["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    original_params = count_params(model)
    original_size = model_size_bytes(model)
    print(f"Original params: {original_params:,} ({original_size/1024/1024:.2f} MB FP32)")

    print("\nBaseline evaluation:")
    eval_root = config["eval"]["eval_root"]
    benchmark_names = config["eval"].get("benchmarks", ["lfw", "cfp_fp", "agedb_30"])
    baseline_results = evaluate_all(model, eval_root, benchmark_names, device=str(device))
    baseline_acc = sum(r["accuracy"] for r in baseline_results.values()) / max(len(baseline_results), 1)
    baseline_per_bm = {k: v["accuracy"] for k, v in baseline_results.items()}
    print(f"Baseline avg: {baseline_acc:.4f}")
    write_eval_csv("baseline", baseline_results, os.path.join(args.output_dir, "eval_metrics.csv"))
    write_roc_json("baseline", baseline_results, os.path.join(args.output_dir, "roc_curves.json"))
    write_scores_json("baseline", baseline_results, os.path.join(args.output_dir, "scores.json"))

    ft_lr = args.finetune_lr
    if ft_lr is None:
        ft_lr = config["train"]["optimizer"]["lr"] / 10.0
    config["train"]["optimizer"]["lr"] = ft_lr
    print(f"Fine-tune LR: {ft_lr}")

    example_inputs = torch.randn(1, 3, config["model"]["input_size"], config["model"]["input_size"]).to(device)
    embedding_dim = config["model"]["embedding_dim"]

    history = [{
        "iteration": 0,
        "cumulative_ratio": 0.0,
        "params": original_params,
        "params_fraction": 1.0,
        "size_bytes": original_size,
        "avg_acc": baseline_acc,
        "acc_drop": 0.0,
        "per_benchmark": baseline_per_bm,
        "time_s": 0.0,
    }]
    iteration = 0

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    while True:
        iteration += 1
        t0 = time.time()
        print(f"\n{'='*60}\nIteration {iteration} — pruning {args.step_ratio*100:.0f}% of current params")

        ignored = get_ignored_layers(model, embedding_dim)
        imp = None
        for name in ("GroupMagnitudeImportance", "GroupNormImportance", "MagnitudeImportance"):
            cls = getattr(tp.importance, name, None)
            if cls is not None:
                imp = cls(p=2)
                break
        if imp is None:
            raise RuntimeError("No gradient-free importance class found in torch_pruning")
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            pruning_ratio=args.step_ratio,
            iterative_steps=1,
            ignored_layers=ignored,
            global_pruning=True,
            isomorphic=True,
            round_to=8,
        )
        pruner.step()

        params_now = count_params(model)
        frac = params_now / original_params
        print(f"Params after prune: {params_now:,} ({frac*100:.1f}% of original)")

        print(f"Recalibrating BN ({args.bn_recal_batches} batches)...")
        recalibrate_bn(model, dataloader, device, args.bn_recal_batches)

        print(f"Fine-tuning {args.finetune_epochs} epochs...")
        params = list(model.parameters()) + list(head.parameters())
        optimizer = build_optimizer(params, config)
        for ep in range(args.finetune_epochs):
            loss, acc = train_one_epoch(
                model, head, dataloader, optimizer, scaler, device,
                config["save"].get("log_interval", 100),
            )
            print(f"  ft epoch {ep+1}/{args.finetune_epochs} — loss {loss:.4f}, train_acc {acc:.4f}")

        eval_results = evaluate_all(model, config["eval"]["eval_root"], benchmark_names, device=str(device))
        avg_acc = sum(r["accuracy"] for r in eval_results.values()) / max(len(eval_results), 1)
        per_bm = {k: v["accuracy"] for k, v in eval_results.items()}
        write_eval_csv(f"iter_{iteration}", eval_results, os.path.join(args.output_dir, "eval_metrics.csv"))
        write_roc_json(f"iter_{iteration}", eval_results, os.path.join(args.output_dir, "roc_curves.json"))
        write_scores_json(f"iter_{iteration}", eval_results, os.path.join(args.output_dir, "scores.json"))
        drop = baseline_acc - avg_acc
        elapsed = time.time() - t0

        print(f"Avg acc: {avg_acc:.4f} (drop {drop:+.4f}) — time {elapsed:.0f}s")

        entry = {
            "iteration": iteration,
            "cumulative_ratio": 1.0 - frac,
            "params": params_now,
            "params_fraction": frac,
            "size_bytes": model_size_bytes(model),
            "avg_acc": avg_acc,
            "acc_drop": drop,
            "per_benchmark": per_bm,
            "time_s": elapsed,
        }
        history.append(entry)

        ckpt_path = os.path.join(args.output_dir, f"pruned_iter{iteration:02d}_frac{frac:.3f}.pth")
        torch.save({"model": model, "head": head.state_dict(), "entry": entry}, ckpt_path)
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump({"baseline_acc": baseline_acc, "history": history}, f, indent=2, default=float)
        write_csv(history, benchmark_names, os.path.join(args.output_dir, "history.csv"))

        if drop > args.max_acc_drop:
            print(f"STOP: acc drop {drop:.4f} > max {args.max_acc_drop}")
            break
        if frac < args.min_params_ratio:
            print(f"STOP: params fraction {frac:.4f} < min {args.min_params_ratio}")
            break
        if iteration >= args.max_iterations:
            print(f"STOP: max iterations ({args.max_iterations}) reached")
            break

    print(f"\n{'='*60}\nFinal fine-tune: {args.final_finetune_epochs} epochs" + (" with KD" if args.use_kd else ""))
    params = list(model.parameters()) + list(head.parameters())
    optimizer = build_optimizer(params, config)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.final_finetune_epochs, eta_min=ft_lr * 0.01)

    for ep in range(args.final_finetune_epochs):
        if args.use_kd and teacher is not None:
            loss, acc = train_one_epoch_kd(
                model, teacher, head, dataloader, optimizer, scaler, device,
                args.kd_weight, config["save"].get("log_interval", 100),
            )
        else:
            loss, acc = train_one_epoch(
                model, head, dataloader, optimizer, scaler, device,
                config["save"].get("log_interval", 100),
            )
        scheduler.step()
        print(f"  final ep {ep+1}/{args.final_finetune_epochs} — loss {loss:.4f}, train_acc {acc:.4f}, lr {scheduler.get_last_lr()[0]:.6f}")

    final_results = evaluate_all(model, config["eval"]["eval_root"], benchmark_names, device=str(device))
    final_acc = sum(r["accuracy"] for r in final_results.values()) / max(len(final_results), 1)
    final_per_bm = {k: v["accuracy"] for k, v in final_results.items()}
    write_eval_csv("final", final_results, os.path.join(args.output_dir, "eval_metrics.csv"))
    write_roc_json("final", final_results, os.path.join(args.output_dir, "roc_curves.json"))
    write_scores_json("final", final_results, os.path.join(args.output_dir, "scores.json"))
    final_drop = baseline_acc - final_acc
    print(f"After final fine-tune: {final_acc:.4f} (drop {final_drop:+.4f})")

    final_params = count_params(model)
    final_frac = final_params / original_params
    final_entry = {
        "iteration": iteration + 1,
        "cumulative_ratio": 1.0 - final_frac,
        "params": final_params,
        "params_fraction": final_frac,
        "size_bytes": model_size_bytes(model),
        "avg_acc": final_acc,
        "acc_drop": final_drop,
        "per_benchmark": final_per_bm,
        "time_s": 0.0,
    }
    history.append(final_entry)

    torch.save(
        {"model": model, "head": head.state_dict(), "entry": final_entry},
        os.path.join(args.output_dir, "final_model.pth"),
    )
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump({"baseline_acc": baseline_acc, "history": history}, f, indent=2, default=float)
    write_csv(history, benchmark_names, os.path.join(args.output_dir, "history.csv"))

    passing = [h for h in history if h["acc_drop"] <= args.max_acc_drop]
    best = passing[-1] if passing else history[0]
    print(f"\n{'='*60}\nDone. Max pruning at ≤{args.max_acc_drop*100:.1f}% drop:")
    print(f"  params: {best['params']:,} ({best['params_fraction']*100:.1f}% of original)")
    print(f"  acc:    {best['avg_acc']:.4f} (drop {best['acc_drop']:+.4f})")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"baseline_acc": baseline_acc, "best": best, "history": history}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
