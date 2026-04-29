from test_siglip_hybrid import analyze_image
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS_CSV = ROOT / "labels.csv"
RESULTS_JSON = ROOT / "benchmark_results.json"


def predict_image(image_path: str) -> dict:
    image_path = ROOT / image_path
    return analyze_image(str(image_path))


def normalize(value: str) -> str:
    return (value or "").strip().lower()


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_row(row: dict, prediction: dict) -> dict:
    true_tag = row["primary_tag"].strip()
    true_family = row["family"].strip()
    should_pass = row["expected_should_pass"].strip() == "1"

    pred_top_tag = (prediction.get("top_tag") or "").strip()
    pred_top_tags = prediction.get("top_tags") or []
    pred_top_family = (prediction.get("top_family") or "").strip()

    top1_correct = normalize(pred_top_tag) == normalize(true_tag)
    top3_correct = any(normalize(tag) == normalize(true_tag) for tag in pred_top_tags[:3])
    family_correct = normalize(pred_top_family) == normalize(true_family)

    false_positive = False
    if not should_pass and pred_top_tag and normalize(pred_top_tag) != "none":
        false_positive = True

    return {
        "image_path": row["image_path"],
        "true_tag": true_tag,
        "true_family": true_family,
        "pred_top_tag": pred_top_tag,
        "pred_top_family": pred_top_family,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "family_correct": family_correct,
        "should_pass": should_pass,
        "false_positive": false_positive,
        "notes": row.get("notes", ""),
    }


def load_labels(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_labels(LABELS_CSV)
    results = []
    by_family = defaultdict(list)
    by_tag = defaultdict(list)
    confusion = Counter()

    for row in rows:
        prediction = predict_image(row["image_path"])
        result = evaluate_row(row, prediction)
        results.append(result)

        by_family[row["family"]].append(result)
        by_tag[row["primary_tag"]].append(result)
        confusion[(row["primary_tag"], result["pred_top_tag"] or "None")] += 1

    total = len(results)
    negatives = [r for r in results if not r["should_pass"]]

    summary = {
        "total_images": total,
        "top1_accuracy": safe_div(sum(r["top1_correct"] for r in results), total),
        "top3_recall": safe_div(sum(r["top3_correct"] for r in results), total),
        "family_accuracy": safe_div(sum(r["family_correct"] for r in results), total),
        "hard_negative_false_positive_rate": safe_div(
            sum(r["false_positive"] for r in negatives),
            len(negatives),
        ),
        "per_family": {},
        "per_tag": {},
        "top_confusions": [],
    }

    for family, family_results in by_family.items():
        count = len(family_results)
        summary["per_family"][family] = {
            "count": count,
            "top1_accuracy": safe_div(sum(r["top1_correct"] for r in family_results), count),
            "top3_recall": safe_div(sum(r["top3_correct"] for r in family_results), count),
            "family_accuracy": safe_div(sum(r["family_correct"] for r in family_results), count),
        }

    for tag, tag_results in by_tag.items():
        count = len(tag_results)
        summary["per_tag"][tag] = {
            "count": count,
            "top1_accuracy": safe_div(sum(r["top1_correct"] for r in tag_results), count),
            "top3_recall": safe_div(sum(r["top3_correct"] for r in tag_results), count),
            "family_accuracy": safe_div(sum(r["family_correct"] for r in tag_results), count),
        }

    for (true_tag, pred_tag), count in confusion.most_common(25):
        if normalize(true_tag) != normalize(pred_tag):
            summary["top_confusions"].append(
                {"true_tag": true_tag, "pred_tag": pred_tag, "count": count}
            )

    output = {"summary": summary, "results": results}
    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("=== Benchmark Summary ===")
    print(f"Total images: {summary['total_images']}")
    print(f"Top-1 accuracy: {summary['top1_accuracy']:.3f}")
    print(f"Top-3 recall: {summary['top3_recall']:.3f}")
    print(f"Family accuracy: {summary['family_accuracy']:.3f}")
    print(
        "Hard-negative false positive rate: "
        f"{summary['hard_negative_false_positive_rate']:.3f}"
    )

    if summary["top_confusions"]:
        print("\nTop confusions:")
        for item in summary["top_confusions"][:10]:
            print(f"  {item['true_tag']} -> {item['pred_tag']}: {item['count']}")


if __name__ == "__main__":
    main()
