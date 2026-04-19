import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_optimizer(params, config):
    cfg = config["train"]["optimizer"]
    opt_type = cfg["type"].lower()

    if opt_type == "sgd":
        return optim.SGD(
            params,
            lr=cfg["lr"],
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 5e-4),
        )
    elif opt_type == "adamw":
        return optim.AdamW(
            params,
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.05),
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def build_scheduler(optimizer, config):
    cfg = config["train"]["scheduler"]
    sch_type = cfg["type"].lower()

    if sch_type == "multistep":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["milestones"],
            gamma=cfg.get("gamma", 0.1),
        )
    elif sch_type == "cosine":
        epochs = config["train"]["epochs"]
        warmup = cfg.get("warmup_epochs", 0)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup)
        if warmup > 0:
            warmup_sch = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
            return SequentialLR(optimizer, [warmup_sch, cosine], milestones=[warmup])
        return cosine
    else:
        raise ValueError(f"Unknown scheduler: {sch_type}")
