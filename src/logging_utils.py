from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> str:
    """Append one JSON record to a JSONL log file."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {"timestamp": utc_timestamp(), **record}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, ensure_ascii=False, default=str) + "\n")
    return str(log_path)
