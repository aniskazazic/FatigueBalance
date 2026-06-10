"""Shared feature preprocessing for CSV training and live inference."""
from typing import Any


POSITION_ALIASES = {
    "goalkeeper": "goalkeeper",
    "gk": "goalkeeper",
    "defender": "defender",
    "outside back": "defender",
    "center back": "defender",
    "left back": "defender",
    "right back": "defender",
    "midfielder": "midfielder",
    "center mid": "midfielder",
    "defensive mid": "midfielder",
    "attacking mid": "midfielder",
    "forward": "forward",
    "center forward": "forward",
    "striker": "forward",
    "right wing": "forward",
    "left wing": "forward",
}


def normalize_position(raw: Any) -> str:
    key = str(raw or "midfielder").strip().lower()
    return POSITION_ALIASES.get(key, "midfielder")


def normalize_activity(raw: Any) -> str:
    value = str(raw or "practice").strip().lower()
    if value in {"game", "match", "competition"}:
        return "game"
    if "game" in value or "match" in value:
        return "game"
    return "practice"


def distance_to_km(value: Any, default: float = 5.0) -> float:
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return default
    if distance > 100:
        return distance / 1000.0
    return distance
