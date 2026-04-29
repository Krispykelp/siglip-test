from typing import Any, Dict, List


VALID_FAMILIES = (
    "Athletics",
    "Smarts",
    "Creativity",
    "Vibes",
    "Sociability",
)


def resolve_rewards_from_analysis(analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extracts final reward decisions directly from analyzer output.

    This is the ONLY place rewards should be interpreted.
    The analyzer has already decided:
    - what tag is supported
    - what family it belongs to
    - reward type (direct / family / inferred)
    - reward amount

    We simply convert it into a clean structure.
    """

    rewards: Dict[str, Dict[str, Any]] = {}

    resolved = analysis.get("resolved_rewards", [])

    for r in resolved:
        family = r.get("family")
        tag = r.get("tag")
        reward_type = r.get("reward_type")
        amount = r.get("amount", 0)

        if not family:
            continue

        rewards[family] = {
            "type": reward_type,     # direct / family / inferred
            "tag": tag,
            "delta": amount,
            "source": "analyzer"
        }

    return rewards


def extract_support_breakdown(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts useful debug info about support classification.
    """

    support = analysis.get("support", {})

    return {
        "direct_supported": support.get("direct_supported", []),
        "family_supported": support.get("family_supported", []),
        "unsupported": support.get("unsupported", []),
        "primary_inferred": support.get("primary_inferred"),
        "bonus_inferred": support.get("bonus_inferred", []),
    }


def build_reward_summary_from_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function used by run_test.py and later backend.

    Converts full analyzer output → clean reward summary.
    """

    claimed_tags = analysis.get("claimed_tags", [])
    top_family = analysis.get("top_family")

    rewards = resolve_rewards_from_analysis(analysis)

    total_delta = sum(r.get("delta", 0) for r in rewards.values())

    per_family_delta = {family: 0 for family in VALID_FAMILIES}
    for family, payload in rewards.items():
        per_family_delta[family] = payload.get("delta", 0)

    return {
        "claimed_tags": claimed_tags,
        "top_family": top_family,

        # ✅ core reward output
        "rewards": rewards,
        "total_delta": total_delta,
        "per_family_delta": per_family_delta,

        # ✅ debug / explainability (VERY useful)
        "support_breakdown": extract_support_breakdown(analysis),
        "family_decision": analysis.get("family_stage", {}).get("decision_summary"),
        "tag_decision": analysis.get("tag_stage", {}).get("decision_summary"),
    }