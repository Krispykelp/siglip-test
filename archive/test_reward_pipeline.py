from analyzer.reward_pipeline import build_reward_summary_from_analysis

analysis = {
    "claimed_tags": ["Basketball", "Workout"],
    "top_family": "Athletics",
    "resolved_rewards": [
        {
            "family": "Athletics",
            "tag": "Basketball",
            "reward_type": "direct",
            "amount": 2,
        }
    ],
    "support": {
        "direct_supported": [["Basketball", -3.99]],
        "family_supported": [["Workout", -22.95]],
        "unsupported": [],
        "primary_inferred": None,
        "bonus_inferred": [],
    },
    "family_stage": {
        "decision_summary": {
            "winner": "Athletics",
            "runner_up": "Social",
        }
    },
    "tag_stage": {
        "decision_summary": {
            "winner": "Basketball",
            "runner_up": "Soccer",
        }
    },
}

result = build_reward_summary_from_analysis(analysis)
print(result)