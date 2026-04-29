from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "benchmark_results.json"
OUTPUT_PATH = ROOT / "tuning_report.txt"

RULES = {
    ("Drawing", "Music"): "Drawing is collapsing into Music. Lower music_like default confidence and raise drawing_like person/creative-focus weights.",
    ("Painting", "Music"): "Painting is collapsing into Music. Raise painting_like scene weights and make Music require stronger people/performance context.",
    ("Photography", "Music"): "Photography is collapsing into Music. Increase photography_like phone/subject capture cues and reduce music_like on non-performance scenes.",
    ("Travel", "Music"): "Travel is falling into Creativity. Boost vibes_family default outdoor/scenery support and strengthen travel_like for landmark/cityscape scenes.",
    ("Workout", "Music"): "Workout is falling into Creativity. Increase workout_like person/activity support and add stronger penalties to creativity for obvious body-activity scenes.",
    ("Hiking", "Music"): "Hiking is falling into Creativity. Increase hiking_like outdoor-person bias and push Vibes/Athletics above Creativity on scenic activity shots.",
    ("Rock Climbing", "Music"): "Rock Climbing is falling into Creativity. Add stronger climbing_like activity-focus support and consider detector/scene heuristics for climbing walls.",
    ("Running", "Party"): "Running is still falling into Social. Further nerf social_family and party_like when no dining/cake/social props are present.",
    ("Chess", "Party"): "Chess is still falling into Social. Add explicit chess-like support and penalize Party/Hangout when the scene is tabletop-focused without party props.",
    ("Basketball", "Support"): "Basketball court/group shots still leak into Social. Increase athletics group-sports bias and lower support/friends on sports scenes.",
}

FAMILY_RECOMMENDATIONS = {
    "Athletics": "Athletics family still underperforms. Prioritize Running, Hiking, Workout, and Rock Climbing with stronger scene heuristics or detector anchors.",
    "Smarts": "Smarts is currently the most stable non-creativity family. Use it as the template for other families.",
    "Creativity": "Creativity family routing is strong now. The problem is intra-family separation, especially Music dominating Drawing/Painting/Photography.",
    "Social": "Social family still steals some examples. Keep tightening tag-level cues so Party/Hangout/Support need clearer social evidence.",
    "Vibes": "Vibes is still failing completely. Add stronger scenic/travel/nature priors so it can beat Creativity on landscapes and landmarks.",
}


def load_results() -> dict:
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_report(data: dict) -> str:
    summary = data["summary"]
    lines: list[str] = []

    lines.append("TUNING REPORT")
    lines.append("=" * 32)
    lines.append("")
    lines.append(f"Total images: {summary['total_images']}")
    lines.append(f"Top-1 accuracy: {summary['top1_accuracy']:.3f}")
    lines.append(f"Top-3 recall: {summary['top3_recall']:.3f}")
    lines.append(f"Family accuracy: {summary['family_accuracy']:.3f}")
    lines.append("")

    lines.append("Family observations")
    lines.append("-" * 20)
    per_family = summary.get("per_family", {})
    for family, info in sorted(per_family.items(), key=lambda kv: kv[1]["family_accuracy"]):
        rec = FAMILY_RECOMMENDATIONS.get(family, "No family-specific recommendation available.")
        lines.append(
            f"{family}: family_acc={info['family_accuracy']:.3f}, top1={info['top1_accuracy']:.3f}, top3={info['top3_recall']:.3f}"
        )
        lines.append(f"  Recommendation: {rec}")
    lines.append("")

    lines.append("Top confusion pairs")
    lines.append("-" * 19)
    for item in summary.get("top_confusions", [])[:15]:
        pair = (item["true_tag"], item["pred_tag"])
        lines.append(f"{item['true_tag']} -> {item['pred_tag']}: {item['count']}")
        lines.append(f"  Recommendation: {RULES.get(pair, 'Review example images and add more tag-specific scene cues for this pair.')}")
    lines.append("")

    results = data.get("results", [])
    family_drifts: dict[str, Counter] = defaultdict(Counter)
    for row in results:
        if not row.get("family_correct"):
            family_drifts[row["true_family"]][row.get("pred_top_family", "None")] += 1

    lines.append("Family drift matrix")
    lines.append("-" * 18)
    for true_family, counter in sorted(family_drifts.items()):
        drift_summary = ", ".join(f"{pred}:{count}" for pred, count in counter.most_common()) or "none"
        lines.append(f"{true_family} -> {drift_summary}")
    lines.append("")

    problematic_tags = []
    for tag, info in summary.get("per_tag", {}).items():
        if info["count"] >= 2 and info["top1_accuracy"] < 0.34:
            problematic_tags.append((tag, info))
    problematic_tags.sort(key=lambda kv: (kv[1]["top1_accuracy"], kv[1]["top3_recall"]))

    lines.append("Priority tags to tune next")
    lines.append("-" * 26)
    for tag, info in problematic_tags[:12]:
        lines.append(
            f"{tag}: top1={info['top1_accuracy']:.3f}, top3={info['top3_recall']:.3f}, family={info['family_accuracy']:.3f}, count={info['count']}"
        )
    lines.append("")

    lines.append("Suggested next moves")
    lines.append("-" * 20)
    lines.append("1. Improve Vibes family routing so Travel/Nature can beat Creativity on scenic images.")
    lines.append("2. Add intra-creativity separation so Drawing/Painting/Photography do not collapse into Music.")
    lines.append("3. Add stronger athletics lifestyle cues for Running, Hiking, Workout, and Rock Climbing.")
    lines.append("4. Keep reducing Social tag-level dominance for Party/Support/Hangout when explicit social props are missing.")

    return "\n".join(lines) + "\n"


def main() -> None:
    data = load_results()
    report = build_report(data)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
