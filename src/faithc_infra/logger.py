from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, event: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            **fields,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def info(self, event: str, **fields: Any) -> None:
        self.log("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self.log("WARNING", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.log("ERROR", event, **fields)
