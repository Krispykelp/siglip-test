from .config import (
    BONUS_MARGIN_FROM_TOP,
    DETECTOR_OVERRIDE_FAMILIES,
    DIRECT_REWARD,
    FAMILY_MIN_MARGIN,
    FAMILY_MIN_SCORE,
    FAMILY_SUPPORT_MAX_DROP_FROM_DIRECT,
    FAMILY_SUPPORT_MARGIN_FROM_TOP,
    FAMILY_SUPPORT_REWARD,
    MAX_BONUS_TAGS,
    MIN_BONUS_SCORE,
    MIN_SUPPORT_SCORE,
    BONUS_REWARD,
    PRIMARY_INFERRED_REWARD,
    SUPPORT_MARGIN_FROM_TOP,
)
from .context import infer_context_signals
from .family_specs import FAMILY_SPECS
from .modules import apply_context_weight_map, apply_module, apply_weight_map
from .tag_specs import FAMILY_TO_STAT, TAG_SPECS, TAG_TO_FAMILY


def build_family_evidence(family, base_score, detection_counts, ctx):
    spec = FAMILY_SPECS[family]
    adjustments = []
    adjusted = base_score

    adjusted = apply_weight_map(
        adjusted, adjustments,
        spec.get("detector_weights", {}),
        detection_counts,
        "detector"
    )

    for module_name in spec.get("modules", []):
        adjusted = apply_module(
            module_name, adjusted, adjustments,
            detection_counts, ctx,
            entity_name=family, entity_type="family"
        )

    return {
        "family": family,
        "base_score": base_score,
        "final_score": adjusted,
        "best_prompt": None,
        "adjustments": adjustments
    }


def summarize_family_decision(family_results_with_evidence, forced_family):
    if not family_results_with_evidence:
        return {
            "winner": None,
            "runner_up": None,
            "margin": None,
            "forced_family": forced_family,
            "reason": "no_family_results"
        }

    winner = family_results_with_evidence[0]
    runner_up = family_results_with_evidence[1] if len(family_results_with_evidence) > 1 else None
    margin = None if runner_up is None else winner["final_score"] - runner_up["final_score"]

    if forced_family is not None:
        reason = f"detector override selected {forced_family}"
    elif runner_up is None:
        reason = "only one viable family"
    else:
        reason = f"{winner['family']} beat {runner_up['family']} by {margin:.3f}"

    return {
        "winner": winner["family"],
        "runner_up": runner_up["family"] if runner_up else None,
        "margin": margin,
        "forced_family": forced_family,
        "reason": reason
    }


def detection_override_family(detection_counts, ctx):
    if detection_counts.get("sports ball", 0) > 0 and detection_counts.get("person", 0) >= 1:
        return "Athletics"
    if detection_counts.get("bicycle", 0) > 0:
        return "Athletics"
    if detection_counts.get("surfboard", 0) > 0:
        return "Athletics"
    if detection_counts.get("skateboard", 0) > 0:
        return "Athletics"
    if detection_counts.get("tennis racket", 0) > 0:
        return "Athletics"
    if detection_counts.get("baseball bat", 0) > 0 or detection_counts.get("baseball glove", 0) > 0:
        return "Athletics"
    if detection_counts.get("skis", 0) > 0 or detection_counts.get("snowboard", 0) > 0:
        return "Athletics"

    if detection_counts.get("laptop", 0) >= 1 and detection_counts.get("person", 0) >= 1:
        return "Smarts"
    if detection_counts.get("book", 0) >= 2 and detection_counts.get("person", 0) >= 1:
        return "Smarts"

    if ctx["structured_social_scene"] and not ctx["sports_scene"] and not ctx["study_scene"]:
        return "Social"
    if ctx["large_group_scene"] and not ctx["sports_scene"] and not ctx["study_scene"]:
        return "Social"
    if detection_counts.get("cake", 0) > 0 and detection_counts.get("person", 0) >= 2:
        return "Social"

    if ctx["pet_scene"] and detection_counts.get("person", 0) <= 1:
        return "Vibes"
    if ctx["travel_scene"] and detection_counts.get("person", 0) <= 2 and not ctx["sports_scene"]:
        return "Vibes"

    return None


def apply_family_fusion(family_results, detection_counts):
    ctx = infer_context_signals(detection_counts)
    traced = []

    for family, score, best_prompt in family_results:
        evidence = build_family_evidence(
            family=family,
            base_score=score,
            detection_counts=detection_counts,
            ctx=ctx
        )
        evidence["best_prompt"] = best_prompt
        traced.append(evidence)

    traced.sort(key=lambda x: x["final_score"], reverse=True)

    forced_family = detection_override_family(detection_counts, ctx)
    if forced_family is not None:
        forced_row = None
        others = []
        for row in traced:
            if row["family"] == forced_family:
                forced_row = row
            else:
                others.append(row)
        if forced_row is not None:
            traced = [forced_row] + others

    compact = [(row["family"], row["final_score"], row["best_prompt"]) for row in traced]
    decision_summary = summarize_family_decision(traced, forced_family)
    return compact, traced, ctx, decision_summary


def family_confidence_passes(family_results, forced_family):
    if forced_family in DETECTOR_OVERRIDE_FAMILIES:
        return True, f"detector_override:{forced_family}"

    if len(family_results) == 0:
        return False, "no_family_results"

    top_family, top_score, _ = family_results[0]
    second_score = family_results[1][1] if len(family_results) > 1 else float("-inf")

    if top_score < FAMILY_MIN_SCORE:
        return False, f"top_family_score_too_low:{top_family}:{top_score:.6f}"

    if len(family_results) > 1 and (top_score - second_score) < FAMILY_MIN_MARGIN:
        return False, f"top_family_margin_too_small:{top_family}:{(top_score - second_score):.6f}"

    return True, "score_margin_pass"


def build_tag_evidence(tag, base_score, detection_counts, ctx):
    spec = TAG_SPECS[tag]
    adjustments = []
    adjusted = base_score

    adjusted = apply_weight_map(
        adjusted, adjustments,
        spec.get("detector_weights", {}),
        detection_counts,
        "detector"
    )

    adjusted = apply_context_weight_map(
        adjusted, adjustments,
        spec.get("context_weights", {}),
        ctx,
        "ctx"
    )

    for module_name in spec.get("modules", []):
        adjusted = apply_module(
            module_name, adjusted, adjustments,
            detection_counts, ctx,
            entity_name=tag, entity_type="tag"
        )

    return {
        "tag": tag,
        "family": spec["family"],
        "base_score": base_score,
        "final_score": adjusted,
        "best_prompt": None,
        "adjustments": adjustments
    }


def summarize_tag_decision(traced_tag_results):
    if not traced_tag_results:
        return {
            "winner": None,
            "runner_up": None,
            "margin": None,
            "reason": "no_tag_results"
        }

    winner = traced_tag_results[0]
    runner_up = traced_tag_results[1] if len(traced_tag_results) > 1 else None
    margin = None if runner_up is None else winner["final_score"] - runner_up["final_score"]

    if runner_up is None:
        reason = "only one viable tag"
    else:
        reason = f"{winner['tag']} beat {runner_up['tag']} by {margin:.3f}"

    return {
        "winner": winner["tag"],
        "runner_up": runner_up["tag"] if runner_up else None,
        "margin": margin,
        "reason": reason
    }


def apply_tag_fusion(canonical_results, detection_counts):
    ctx = infer_context_signals(detection_counts)
    traced = []

    for tag, score, best_prompt in canonical_results:
        evidence = build_tag_evidence(
            tag=tag,
            base_score=score,
            detection_counts=detection_counts,
            ctx=ctx
        )
        evidence["best_prompt"] = best_prompt
        traced.append(evidence)

    traced.sort(key=lambda x: x["final_score"], reverse=True)

    compact = [(row["tag"], row["final_score"], row["best_prompt"]) for row in traced]
    decision_summary = summarize_tag_decision(traced)
    return compact, traced, decision_summary


def is_supported_claimed(score, top_score):
    return score >= MIN_SUPPORT_SCORE and (top_score - score) <= SUPPORT_MARGIN_FROM_TOP


def is_family_supported_claimed(claimed_tag, top_family, score, top_score, best_direct_score):
    tag_family = TAG_TO_FAMILY.get(claimed_tag)
    if tag_family != top_family:
        return False
    if score is None:
        return False
    if (top_score - score) > FAMILY_SUPPORT_MARGIN_FROM_TOP:
        return False
    if best_direct_score is not None and (best_direct_score - score) > FAMILY_SUPPORT_MAX_DROP_FROM_DIRECT:
        return False
    return True


def select_primary_and_bonus_inferred(canonical_results, claimed_tags, top_family):
    primary_inferred = None
    bonus_inferred = []

    if not canonical_results:
        return primary_inferred, bonus_inferred

    top_tag, top_score, top_prompt = canonical_results[0]

    if TAG_TO_FAMILY.get(top_tag) != top_family:
        return primary_inferred, bonus_inferred

    if len(claimed_tags) == 0:
        primary_inferred = (top_tag, top_score, top_prompt)

        for tag, score, best_prompt in canonical_results[1:]:
            if TAG_TO_FAMILY.get(tag) != top_family:
                continue
            if score < MIN_BONUS_SCORE:
                continue
            if (top_score - score) > BONUS_MARGIN_FROM_TOP:
                continue
            bonus_inferred.append((tag, score, best_prompt))
    else:
        for tag, score, best_prompt in canonical_results:
            if tag in claimed_tags:
                continue
            if TAG_TO_FAMILY.get(tag) != top_family:
                continue
            if score < MIN_BONUS_SCORE:
                continue
            if (top_score - score) > BONUS_MARGIN_FROM_TOP:
                continue
            bonus_inferred.append((tag, score, best_prompt))

    return primary_inferred, bonus_inferred[:MAX_BONUS_TAGS]


def build_verifier_summary(canonical_results, claimed_tags, top_family):
    top_score = canonical_results[0][1] if canonical_results else float("-inf")
    score_lookup = {tag: score for tag, score, _ in canonical_results}

    direct_supported = []
    family_supported = []
    unsupported = []

    for claimed in claimed_tags:
        score = score_lookup.get(claimed)
        if score is None:
            unsupported.append((claimed, None))
        elif is_supported_claimed(score, top_score):
            direct_supported.append((claimed, score))
        else:
            unsupported.append((claimed, score))

    best_direct_score = max((score for _, score in direct_supported), default=None)

    remaining = []
    for claimed, score in unsupported:
        if score is not None and is_family_supported_claimed(
            claimed_tag=claimed,
            top_family=top_family,
            score=score,
            top_score=top_score,
            best_direct_score=best_direct_score
        ):
            family_supported.append((claimed, score))
        else:
            remaining.append((claimed, score))

    unsupported = remaining
    primary_inferred, bonus_inferred = select_primary_and_bonus_inferred(
        canonical_results=canonical_results,
        claimed_tags=claimed_tags,
        top_family=top_family
    )

    return direct_supported, family_supported, unsupported, primary_inferred, bonus_inferred


def resolve_rewards(direct_supported, family_supported, primary_inferred, bonus_inferred):
    family_best = {}

    def try_assign(tag, score, reward_type, amount):
        family = TAG_TO_FAMILY.get(tag)
        if family is None:
            return

        candidate = {
            "family": family,
            "stat": FAMILY_TO_STAT[family],
            "tag": tag,
            "score": score,
            "reward_type": reward_type,
            "amount": amount
        }

        current = family_best.get(family)
        priority = {
            "direct": 4,
            "family": 3,
            "primary_inferred": 2,
            "bonus": 1
        }

        if current is None:
            family_best[family] = candidate
            return

        current_pri = priority[current["reward_type"]]
        candidate_pri = priority[candidate["reward_type"]]

        if candidate_pri > current_pri:
            family_best[family] = candidate
        elif candidate_pri == current_pri and candidate["score"] > current["score"]:
            family_best[family] = candidate

    for tag, score in direct_supported:
        try_assign(tag, score, "direct", DIRECT_REWARD)

    for tag, score in family_supported:
        try_assign(tag, score, "family", FAMILY_SUPPORT_REWARD)

    if primary_inferred is not None:
        tag, score, _ = primary_inferred
        try_assign(tag, score, "primary_inferred", PRIMARY_INFERRED_REWARD)

    for tag, score, _ in bonus_inferred:
        try_assign(tag, score, "bonus", BONUS_REWARD)

    return list(family_best.values())