import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64.0, margin_warmup_epochs=0):
        super().__init__()
        self.target_margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        self.scale = scale

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        start = 0.0 if margin_warmup_epochs > 0 else margin
        self._set_margin(start)

    def _set_margin(self, m):
        self.margin = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def set_epoch(self, epoch):
        if self.margin_warmup_epochs <= 0:
            return
        ratio = min(1.0, (epoch + 1) / self.margin_warmup_epochs)
        self._set_margin(self.target_margin * ratio)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)

        cos_theta = F.linear(embeddings, weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta = torch.sqrt((1.0 - cos_theta.pow(2)).clamp(min=0.0))

        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.long().unsqueeze(1), 1.0)

        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        return logits * self.scale
