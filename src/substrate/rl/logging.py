import csv
import os
from typing import Any, Dict


class CsvLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fieldnames = None
        self._file = open(path, "w", newline="")
        self._writer = None

    def log(self, step: int, metrics: Dict[str, Any]):
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()
