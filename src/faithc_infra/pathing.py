from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional


class PathManager:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def resolve_path(value: str, base_dir: Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
        return slug.strip("-") or "sample"

    def create_run_dir(self, prefix: str, run_id: Optional[str] = None) -> Path:
        rid = run_id or f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir = self.runs_dir / rid
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def create_sample_dir(self, run_dir: Path, sample_name: str) -> Path:
        sample_dir = run_dir / self._slugify(sample_name)
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir
