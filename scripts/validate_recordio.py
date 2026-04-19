#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

from data.recordio import IndexedRecordIO
from data.recordio import unpack
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Validate MXNet RecordIO dataset and cache valid indices.")
    parser.add_argument("--data_root", default="data/ms1mv3")
    parser.add_argument("--cache_name", default="valid_indices.pkl")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of records to scan (0 = all).")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    idx_path = data_root / "train.idx"
    rec_path = data_root / "train.rec"
    cache_path = data_root / args.cache_name

    record = IndexedRecordIO(str(idx_path), str(rec_path))

    header0, _ = unpack(record.read_idx(0))
    max_image_idx = int(header0.label[0]) if header0.flag > 0 else max(record.keys) + 1

    image_keys = [k for k in record.keys if 0 < k < max_image_idx]

    valid_indices = []
    bad_records = []
    total = len(image_keys)
    to_scan = total if args.limit <= 0 else min(total, args.limit)

    for n, idx in enumerate(image_keys[:to_scan], start=1):
        try:
            raw = record.read_idx(int(idx))
            header, img_bytes = unpack(raw)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            valid_indices.append(int(idx))
        except Exception as e:
            bad_records.append((int(idx), str(e)))
            print(f"BAD idx={idx}: {e}")

        if n % 500000 == 0:
            print(f"Scanned {n}/{to_scan}...")

    record.close()

    with open(cache_path, "wb") as f:
        pickle.dump(valid_indices, f)

    print()
    print(f"Scanned: {to_scan}/{total}")
    print(f"Valid:   {len(valid_indices)}")
    print(f"Bad:     {len(bad_records)}")
    print(f"Cache:   {cache_path}")

    if bad_records:
        print("\nFirst bad records:")
        for idx, err in bad_records[:20]:
            print(f"  idx={idx}: {err}")


if __name__ == "__main__":
    main()
