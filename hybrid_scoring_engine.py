from typing import Dict, Any

# Bridge to your archived working scorer
from archive.test_siglip import analyze_image as legacy_analyze_image


def apply_calibration_to_result(
    result: Dict[str, Any],
    calibration: Dict[str, Any] | None,
    family_bias_weight: float = 0.15,
    tag_bias_weight: float = 0.0,
) -> Dict[str, Any]:
    if calibration is None:
        return result

    family_bias = calibration.get("family_bias", {})
    tag_bias = calibration.get("tag_bias", {})

    top_family = result.get("top_family", "None")
    top_tag = result.get("top_tag", "None")
    top_tags = list(result.get("top_tags", []))

    family_shift = max(min(family_bias.get(top_family, 0.0), 1.0), -1.0) * family_bias_weight
    tag_shift = max(min(tag_bias.get(top_tag, 0.0), 0.5), -0.5) * tag_bias_weight

    # Conservative calibration behavior:
    # do not invent a new winner without raw score tables
    if top_tag == "None" and top_tags:
        non_none = [t for t in top_tags if t != "None"]
        if non_none:
            result["top_tag"] = non_none[0]

    _ = family_shift
    _ = tag_shift

    return result


def score_image(
    image_path: str,
    calibration: Dict[str, Any] | None = None,
    family_bias_weight: float = 0.15,
    tag_bias_weight: float = 0.0,
) -> Dict[str, Any]:
    result = legacy_analyze_image(image_path)

    if "top_tag" not in result:
        raise RuntimeError("Legacy analyzer did not return expected keys.")

    result = apply_calibration_to_result(
        result=result,
        calibration=calibration,
        family_bias_weight=family_bias_weight,
        tag_bias_weight=tag_bias_weight,
    )

    return result