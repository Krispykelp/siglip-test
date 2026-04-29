import json
from pathlib import Path
from typing import Dict, List

_MAP_PATH = Path(__file__).resolve().parent / "tag_family_map.v1.json"

VALID_FAMILIES = {
    "Athletics",
    "Smarts",
    "Creativity",
    "Vibes",
    "Sociability",
}


def _load_tag_family_map() -> Dict[str, str]:
    if not _MAP_PATH.exists():
        raise FileNotFoundError(f"Tag family map not found at: {_MAP_PATH}")

    with _MAP_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "map" in data:
        data = data["map"]

    if not isinstance(data, dict):
        raise ValueError("tag_family_map.v1.json must contain a JSON object of tag -> family")

    for tag, family in data.items():
        if not isinstance(tag, str) or not isinstance(family, str):
            raise ValueError("Each tag mapping must be a string -> string pair")
        if family not in VALID_FAMILIES:
            raise ValueError(f"Invalid family '{family}' for tag '{tag}'")

    return data

TAG_FAMILY_MAP: Dict[str, str] = _load_tag_family_map()

# lowercase lookup for safety
_TAG_LOOKUP_LOWER = {tag.lower(): tag for tag in TAG_FAMILY_MAP.keys()}


def get_all_known_tags() -> List[str]:
    return list(TAG_FAMILY_MAP.keys())


def normalize_claimed_tag(tag: str) -> str | None:
    if not tag or not isinstance(tag, str):
        return None

    cleaned = tag.strip()
    if not cleaned:
        return None

    canonical = _TAG_LOOKUP_LOWER.get(cleaned.lower())
    return canonical


def get_family_for_tag(tag: str) -> str | None:
    canonical = normalize_claimed_tag(tag)
    if canonical is None:
        return None
    return TAG_FAMILY_MAP.get(canonical)


def map_claimed_tags_to_families(tags: List[str]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {
        "Athletics": [],
        "Smarts": [],
        "Creativity": [],
        "Vibes": [],
        "Sociability": [],
    }

    seen = set()

    for raw_tag in tags:
        canonical = normalize_claimed_tag(raw_tag)
        if canonical is None:
            print(f"[WARN] Unmapped claimed tag: {raw_tag}")
            continue

        if canonical in seen:
            continue

        seen.add(canonical)

        family = TAG_FAMILY_MAP[canonical]
        result[family].append(canonical)

    return result