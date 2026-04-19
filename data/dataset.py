import numbers
import os
import pickle
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.recordio import IndexedRecordIO, unpack


class RecordIODataset(Dataset):

    def __init__(self, data_root, subset_ids=None, k_per_id=None, seed=42):
        path_idx = os.path.join(data_root, "train.idx")
        path_rec = os.path.join(data_root, "train.rec")
        prop_path = os.path.join(data_root, "property")

        missing = [
            path for path in (path_idx, path_rec, prop_path) if not os.path.exists(path)
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(
                f"Missing RecordIO dataset files in '{data_root}': {missing_str}"
            )

        self.record = IndexedRecordIO(path_idx, path_rec)

        header, _ = unpack(self.record.read_idx(0))
        if header.flag > 0:
            self.imgidx = np.arange(1, int(header.label[0]))
        else:
            self.imgidx = np.array(list(self.record.keys))

        with open(prop_path, "r") as f:
            self.num_classes = int(f.read().strip().split(",")[0])

        self.subset_labels = None
        self.valid_indices = None
        if k_per_id is not None:
            self._apply_per_identity_subset(data_root, k_per_id, seed)
        elif subset_ids is not None:
            self._apply_subset(data_root, subset_ids, seed)

        valid_cache = os.path.join(data_root, "valid_indices.pkl")
        if os.path.exists(valid_cache):
            with open(valid_cache, "rb") as f:
                valid_indices = pickle.load(f)
            valid_set = set(int(x) for x in valid_indices)
            mask = np.array([int(idx) in valid_set for idx in self.imgidx])
            self.imgidx = self.imgidx[mask]
            if self.subset_labels is not None:
                self.subset_labels = self.subset_labels[mask]
            self.valid_indices = valid_set
            print(f"Using cached valid indices: {len(self.imgidx)} records")

    @staticmethod
    def _parse_label(header):
        label = header.label
        return int(label) if isinstance(label, numbers.Number) else int(label[0])

    def _load_label_cache(self, data_root):
        cache_path = os.path.join(data_root, "labels_cache.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print("First run: scanning labels (takes a few minutes)...")
        all_labels = np.zeros(len(self.imgidx), dtype=np.int64)
        for i, idx in enumerate(self.imgidx):
            header, _ = unpack(self.record.read_idx(int(idx)))
            all_labels[i] = self._parse_label(header)
            if (i + 1) % 500000 == 0:
                print(f"  {i + 1}/{len(self.imgidx)}")
        with open(cache_path, "wb") as f:
            pickle.dump(all_labels, f)
        print("Label cache saved.")
        return all_labels

    def _apply_subset(self, data_root, subset_ids, seed):
        all_labels = self._load_label_cache(data_root)

        unique_ids = np.unique(all_labels)
        rng = random.Random(seed)
        selected = set(rng.sample(list(unique_ids), min(subset_ids, len(unique_ids))))
        mask = np.isin(all_labels, list(selected))

        self.imgidx = self.imgidx[mask]
        self.subset_labels = all_labels[mask]

        label_map = {old: new for new, old in enumerate(sorted(selected))}
        self.subset_labels = np.array([label_map[l] for l in self.subset_labels])
        self.num_classes = len(selected)

    def _apply_per_identity_subset(self, data_root, k_per_id, seed):
        all_labels = self._load_label_cache(data_root)

        by_label = defaultdict(list)
        for i in range(len(all_labels)):
            by_label[int(all_labels[i])].append(i)

        rng = random.Random(seed)
        selected = []
        for positions in by_label.values():
            if len(positions) <= k_per_id:
                selected.extend(positions)
            else:
                selected.extend(rng.sample(positions, k_per_id))

        selected = np.array(sorted(selected))
        self.imgidx = self.imgidx[selected]
        print(f"Per-identity subset: {len(self.imgidx)} images across {self.num_classes} identities")

    def __len__(self):
        return len(self.imgidx)

    def __getitem__(self, index):
        # Some RecordIO entries can be corrupted; retry a few nearby samples
        # so a single bad record does not kill the whole training run.
        for offset in range(100):
            sample_index = (index + offset) % len(self.imgidx)
            idx = int(self.imgidx[sample_index])
            try:
                record = self.record.read_idx(idx)
                header, img_bytes = unpack(record)

                if self.subset_labels is not None:
                    label = int(self.subset_labels[sample_index])
                else:
                    label = self._parse_label(header)

                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image at idx {idx}")
                img = np.transpose(img, (2, 0, 1)).astype(np.float32)
                img = (img - 127.5) / 127.5

                return torch.from_numpy(img), label
            except Exception as e:
                if offset == 99:
                    raise ValueError(f"Failed to load record after retries, last idx {idx}: {e}") from e

    def __del__(self):
        if hasattr(self, "record"):
            self.record.close()


def build_dataloader(config):
    data_cfg = config["data"]
    train_cfg = config["train"]

    k_per_id = data_cfg.get("k_per_id")
    subset_ids = data_cfg.get("subset_ids") if data_cfg.get("subset") else None

    dataset = RecordIODataset(
        data_root=data_cfg["root"],
        subset_ids=subset_ids,
        k_per_id=k_per_id,
        seed=train_cfg.get("seed", 42),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=True,
        persistent_workers=train_cfg.get("num_workers", 8) > 0,
        drop_last=True,
    )

    return dataloader, dataset.num_classes
