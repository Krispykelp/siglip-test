import json
from hybrid_scoring_engine import score_image

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
    calibration = get_calibration()

    result = score_image(
        image_path=image_path,
        calibration=calibration,
        family_bias_weight=FAMILY_BIAS_WEIGHT,
        tag_bias_weight=TAG_BIAS_WEIGHT,
    )

    return {
        "top_tag": result["top_tag"],
        "top_tags": result["top_tags"],
        "top_family": result["top_family"],
    }