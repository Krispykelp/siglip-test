import json
from analyzer.engine import run_analysis
from analyzer.tag_mapper import get_family_for_tag

USE_CALIBRATION = False
FAMILY_BIAS_WEIGHT = 0.15
TAG_BIAS_WEIGHT = 0.0

_calibration_profile = None


def load_calibration():
    global _calibration_profile
    try:
        with open("calibration_profile.json", "r", encoding="utf-8") as f:
            _calibration_profile = json.load(f)
            print("[Calibration] Loaded profile")
    except Exception:
        _calibration_profile = None
        print("[Calibration] No profile found")


def get_calibration():
    if not USE_CALIBRATION:
        return None
    if _calibration_profile is None:
        load_calibration()
    return _calibration_profile


def analyze_image(image_path: str):
    _ = get_calibration()
    _ = FAMILY_BIAS_WEIGHT
    _ = TAG_BIAS_WEIGHT

    result = run_analysis(image_path=image_path, claimed_tags=[])

    tag_results = result.get("tag_stage", {}).get("results", [])
    family_results = result.get("family_stage", {}).get("results", [])

    top_tag = result.get("top_tag")
    if not top_tag:
        top_tag = tag_results[0][0] if tag_results else "None"

    top_tags = result.get("top_tags")
    if not top_tags:
        top_tags = [row[0] for row in tag_results[:3]]

    top_family = None
    if top_tag and top_tag != "None":
        top_family = get_family_for_tag(top_tag)

    if not top_family:
        top_family = result.get("top_family")
    if not top_family:
        top_family = family_results[0][0] if family_results else "None"

    return {
        "top_tag": top_tag or "None",
        "top_tags": top_tags,
        "top_family": top_family or "None",
    }