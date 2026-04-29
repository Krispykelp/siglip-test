def make_analysis_result(
    image_path,
    claimed_tags,
    detections_all,
    detections_trusted,
    detection_time,
    family_results,
    traced_family_results,
    family_time,
    family_gate_ok,
    family_gate_reason,
    context_signals,
    family_decision_summary,
    canonical_results,
    traced_tag_results,
    tag_decision_summary,
    tag_time,
    direct_supported,
    family_supported,
    unsupported,
    primary_inferred,
    bonus_inferred,
    rewards,
):
    return {
        "status": "verified" if family_gate_ok else "no_strong_match",
        "image_path": image_path,
        "claimed_tags": list(claimed_tags),
        "detections": {
            "all": dict(detections_all),
            "trusted": dict(detections_trusted),
            "stage_time_sec": detection_time,
        },
        "family_stage": {
            "results": family_results,
            "evidence": traced_family_results,
            "stage_time_sec": family_time,
            "gate_pass": family_gate_ok,
            "gate_reason": family_gate_reason,
            "context_signals": context_signals,
            "decision_summary": family_decision_summary,
        },
        "tag_stage": {
            "results": canonical_results,
            "evidence": traced_tag_results,
            "decision_summary": tag_decision_summary,
            "stage_time_sec": tag_time,
        },
        "support": {
            "direct_supported": direct_supported,
            "family_supported": family_supported,
            "unsupported": unsupported,
            "primary_inferred": primary_inferred,
            "bonus_inferred": bonus_inferred,
        },
        "resolved_rewards": rewards,
        "top_family": family_results[0][0] if family_results else None,
    }


def _build_top_supported_tag(result: dict):
    support = result["support"]

    if support["direct_supported"]:
        tag, score = support["direct_supported"][0]
        return {
            "tag": tag,
            "support_type": "direct",
            "score": score,
        }

    if support["family_supported"]:
        tag, score = support["family_supported"][0]
        return {
            "tag": tag,
            "support_type": "same_family",
            "score": score,
        }

    if support["primary_inferred"] is not None:
        tag, score, best_prompt = support["primary_inferred"]
        return {
            "tag": tag,
            "support_type": "primary_inferred",
            "score": score,
            "best_prompt": best_prompt,
        }

    if support["bonus_inferred"]:
        tag, score, best_prompt = support["bonus_inferred"][0]
        return {
            "tag": tag,
            "support_type": "bonus_inferred",
            "score": score,
            "best_prompt": best_prompt,
        }

    return None


def _build_verification_summary(result: dict):
    support = result["support"]

    if result["status"] != "verified":
        return "No strong match was found."

    parts = []

    if support["direct_supported"]:
        direct_names = [tag for tag, _ in support["direct_supported"]]
        parts.append(f"{', '.join(direct_names)} verified directly")

    if support["family_supported"]:
        family_names = [tag for tag, _ in support["family_supported"]]
        parts.append(f"{', '.join(family_names)} supported at family level")

    if support["unsupported"]:
        unsupported_names = [tag for tag, _ in support["unsupported"]]
        parts.append(f"{', '.join(unsupported_names)} unsupported")

    if support["primary_inferred"] is not None:
        tag, _, _ = support["primary_inferred"]
        parts.append(f"{tag} inferred as primary tag")

    if support["bonus_inferred"]:
        bonus_names = [tag for tag, _, _ in support["bonus_inferred"]]
        parts.append(f"{', '.join(bonus_names)} inferred as bonus tags")

    if not parts:
        return "Verified with no explicit supported tags."

    return "; ".join(parts) + "."


def make_compact_analysis_result(result: dict, analysis_id: str | None = None, post_id=None) -> dict:
    family_stage = result["family_stage"]
    support = result["support"]

    compact_rewards = []
    for reward in result["resolved_rewards"]:
        compact_rewards.append({
            "family": reward["family"],
            "stat": reward["stat"],
            "tag": reward["tag"],
            "reward_type": reward["reward_type"],
            "amount": reward["amount"],
            "score": reward["score"],
        })

    direct_supported_tags = [
        {"tag": tag, "score": score}
        for tag, score in support["direct_supported"]
    ]

    same_family_supported_tags = [
        {"tag": tag, "score": score}
        for tag, score in support["family_supported"]
    ]

    unsupported_tags = [
        {"tag": tag, "score": score}
        for tag, score in support["unsupported"]
    ]

    primary_inferred = None
    if support["primary_inferred"] is not None:
        tag, score, best_prompt = support["primary_inferred"]
        primary_inferred = {
            "tag": tag,
            "score": score,
            "best_prompt": best_prompt,
        }

    bonus_inferred_tags = [
        {"tag": tag, "score": score, "best_prompt": best_prompt}
        for tag, score, best_prompt in support["bonus_inferred"]
    ]

    compact = {
        "verification_state": "approved" if result["status"] == "verified" else "no_strong_match",
        "image_path": result["image_path"],
        "claimed_tags": result["claimed_tags"],
        "top_family": result["top_family"],
        "top_supported_tag": _build_top_supported_tag(result),
        "verification_summary": _build_verification_summary(result),
        "family_decision": {
            "winner": family_stage["decision_summary"]["winner"],
            "runner_up": family_stage["decision_summary"]["runner_up"],
            "margin": family_stage["decision_summary"]["margin"],
            "forced_family": family_stage["decision_summary"]["forced_family"],
            "reason": family_stage["decision_summary"]["reason"],
            "gate_pass": family_stage["gate_pass"],
            "gate_reason": family_stage["gate_reason"],
        },
        "direct_supported_tags": direct_supported_tags,
        "same_family_supported_tags": same_family_supported_tags,
        "unsupported_tags": unsupported_tags,
        "primary_inferred_tag": primary_inferred,
        "bonus_inferred_tags": bonus_inferred_tags,
        "reward_events": compact_rewards,
    }

    if analysis_id is not None:
        compact["analysis_id"] = analysis_id

    if post_id is not None:
        compact["post_id"] = post_id

    return compact