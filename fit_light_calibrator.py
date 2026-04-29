import json
from collections import defaultdict

MIN_FAMILY_SAMPLES = 8
MIN_TAG_SAMPLES = 6

def main():
    with open("benchmark_results.json", "r") as f:
        data = json.load(f)

    results = data["results"]

    family_stats = defaultdict(list)
    tag_stats = defaultdict(list)

    for r in results:
        family = r["true_family"]
        tag = r["true_tag"]

        correct = r["top1_correct"]

        family_stats[family].append(correct)
        tag_stats[tag].append(correct)

    calibration = {
        "family_bias": {},
        "tag_bias": {}
    }

    # -------- FAMILY BIAS --------
    for family, vals in family_stats.items():
        if len(vals) < MIN_FAMILY_SAMPLES:
            continue

        acc = sum(vals) / len(vals)

        # center around 0.5 baseline
        bias = (acc - 0.5) * 2

        calibration["family_bias"][family] = round(bias, 3)

    # -------- TAG BIAS (DISABLED FOR NOW) --------
    # Too unstable with small dataset

    with open("calibration_profile.json", "w") as f:
        json.dump(calibration, f, indent=2)

    print("Calibration profile updated (SAFE MODE)")


if __name__ == "__main__":
    main()