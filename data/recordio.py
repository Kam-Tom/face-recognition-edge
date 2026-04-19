import os
import struct
from collections import namedtuple

import numpy as np

Header = namedtuple("Header", ["flag", "label", "id", "id2"])

_REC_MAGIC = 0xCED7230A
_IR_FORMAT = "<IfQQ"
_IR_SIZE = 24


class IndexedRecordIO:

    def __init__(self, idx_path, rec_path):
        self.rec_path = rec_path
        self.index = {}
        with open(idx_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.index[int(parts[0])] = int(parts[1])
        self.keys = list(self.index.keys())
        self._fp = None
        self._fp_pid = None

    def _get_fp(self):
        pid = os.getpid()
        if self._fp is None or self._fp_pid != pid:
            self._fp = open(self.rec_path, "rb")
            self._fp_pid = pid
        return self._fp

    def read_idx(self, idx):
        fp = self._get_fp()
        fp.seek(self.index[idx])
        magic, lrec = struct.unpack("<II", fp.read(8))
        if magic != _REC_MAGIC:
            raise ValueError(f"Bad RecordIO magic at idx {idx}: {magic:#x}")
        length = lrec & ((1 << 29) - 1)
        return fp.read(length)

    def close(self):
        if self._fp is not None and not self._fp.closed:
            self._fp.close()
        self._fp = None

    def __del__(self):
        self.close()


def unpack(s):
    flag, label, id_, id2 = struct.unpack(_IR_FORMAT, s[:_IR_SIZE])
    s = s[_IR_SIZE:]
    if flag > 0:
        label = np.frombuffer(s[: flag * 4], dtype=np.float32).copy()
        s = s[flag * 4 :]
    return Header(flag, label, id_, id2), s
