"""Persist and load ML evaluation metrics for demos and coursework."""
import json
import os
from typing import Any, Dict


def _metrics_path() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, "data", "ml_metrics.json")


def save_metrics(metrics: Dict[str, Any]) -> None:
    path = _metrics_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def load_metrics() -> Dict[str, Any]:
    path = _metrics_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
