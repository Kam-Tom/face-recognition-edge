import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_dataloader
from eval.evaluate import evaluate_all
from losses.arcface import ArcFace
from models import get_model
from train.optimizer import build_optimizer, build_scheduler


def deep_merge(base, override):
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path, data_config_path=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_path = config.pop("_base_", None)
    if base_path:
        base_full = os.path.join(os.path.dirname(config_path), base_path)
        with open(base_full) as f:
            base = yaml.safe_load(f)
        config = deep_merge(base, config)

    if data_config_path:
        with open(data_config_path) as f:
            data_override = yaml.safe_load(f)
        config = deep_merge(config, data_override)

    return config


def model_size_mb(model):
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_total += sum(buf.numel() * buf.element_size() for buf in model.buffers())
    return bytes_total / 1024 / 1024


def write_training_csv(log, benchmark_names, csv_path):
    fieldnames = [
        "epoch", "loss", "train_acc", "lr",
        "params", "size_mb",
        "avg_acc", *benchmark_names, "time_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for entry in log:
            row = {
                "epoch": entry["epoch"],
                "loss": round(entry["loss"], 4),
                "train_acc": round(entry["train_acc"], 4),
                "lr": entry["lr"],
                "params": entry.get("params", 0),
                "size_mb": round(entry.get("size_mb", 0), 3),
                "avg_acc": round(entry.get("avg_eval", 0), 4),
                "time_s": round(entry.get("time", 0), 1),
            }
            for bm in benchmark_names:
                row[bm] = round(entry.get(bm, 0), 4)
            w.writerow(row)


def write_eval_csv(stage, eval_results, csv_path):
    fieldnames = ["stage", "benchmark", "accuracy", "std", "eer", "auc", "tar_far_1e-3", "tar_far_1e-4"]
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
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


def train_one_epoch(model, head, dataloader, optimizer, scaler, device, log_interval, grad_clip=None):
    model.train()
    head.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    params = list(model.parameters()) + list(head.parameters())

    pbar = tqdm(dataloader, desc="Training")
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            embeddings = model(images)
            logits = head(embeddings, labels)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += images.size(0)

        if (i + 1) % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_config", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.data_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config["train"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Loading dataset...")
    dataloader, num_classes = build_dataloader(config)
    print(f"Classes: {num_classes}, Batches: {len(dataloader)}")

    model = get_model(config).to(device)
    head = ArcFace(
        embedding_dim=config["model"]["embedding_dim"],
        num_classes=num_classes,
        margin=config["loss"]["margin"],
        scale=config["loss"]["scale"],
        margin_warmup_epochs=config["loss"].get("margin_warmup_epochs", 0),
    ).to(device)

    params = list(model.parameters()) + list(head.parameters())
    model_param_count = sum(p.numel() for p in model.parameters())
    model_size = model_size_mb(model)
    print(f"Model params: {model_param_count:,} ({model_size:.2f} MB FP32)")

    optimizer = build_optimizer(params, config)
    scheduler = build_scheduler(optimizer, config)

    use_amp = config["train"].get("mixed_precision", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    save_cfg = config["save"]
    model_name = config["model"]["name"]
    output_dir = os.path.join(save_cfg["output_dir"], model_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    log_interval = save_cfg.get("log_interval", 100)
    grad_clip = config["train"].get("grad_clip")
    eval_benchmarks = config["eval"].get("benchmarks", ["lfw", "cfp_fp", "agedb_30"])
    eval_root = config["eval"]["eval_root"]
    best_acc = 0
    log = []

    for epoch in range(start_epoch, config["train"]["epochs"]):
        head.set_epoch(epoch)
        t0 = time.time()
        loss, acc = train_one_epoch(model, head, dataloader, optimizer, scaler, device, log_interval, grad_clip)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1}/{config['train']['epochs']} — loss: {loss:.4f}, acc: {acc:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}, time: {elapsed:.0f}s")

        print("Evaluating...")
        eval_results = evaluate_all(model, eval_root, eval_benchmarks, device=str(device))
        avg_acc = sum(r["accuracy"] for r in eval_results.values()) / max(len(eval_results), 1)

        entry = {
            "epoch": epoch + 1, "loss": loss, "train_acc": acc,
            "lr": optimizer.param_groups[0]["lr"], "time": elapsed,
            "params": model_param_count, "size_mb": model_size,
        }
        for name, res in eval_results.items():
            entry[name] = res["accuracy"]
        entry["avg_eval"] = avg_acc
        log.append(entry)

        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)
        write_training_csv(log, eval_benchmarks, os.path.join(output_dir, "training_log.csv"))
        write_eval_csv(f"epoch_{epoch+1}", eval_results, os.path.join(output_dir, "eval_metrics.csv"))
        write_roc_json(f"epoch_{epoch+1}", eval_results, os.path.join(output_dir, "roc_curves.json"))
        write_scores_json(f"epoch_{epoch+1}", eval_results, os.path.join(output_dir, "scores.json"))

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save({"model": model.state_dict(), "head": head.state_dict(), "epoch": epoch, "accuracy": best_acc}, os.path.join(output_dir, "best_model.pth"))
            print(f"  New best: {best_acc:.4f}")

        torch.save({
            "model": model.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
        }, os.path.join(output_dir, "last_model.pth"))

    print("\nFinal evaluation (best checkpoint):")
    best_ckpt = torch.load(os.path.join(output_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    all_benchmarks = config["eval"].get("benchmarks", ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"])
    final_results = evaluate_all(model, eval_root, all_benchmarks, device=str(device))

    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2, default=float)
    write_eval_csv("final", final_results, os.path.join(output_dir, "eval_metrics.csv"))
    write_roc_json("final", final_results, os.path.join(output_dir, "roc_curves.json"))
    write_scores_json("final", final_results, os.path.join(output_dir, "scores.json"))

    print("Done.")


if __name__ == "__main__":
    main()
