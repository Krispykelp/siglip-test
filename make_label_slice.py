import csv
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "labels.csv"
OUTPUT_DIR = ROOT / "benchmark_slices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows() -> list[dict]:
    with INPUT_FILE.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(rows: List[dict], filename: str) -> None:
    if not rows:
        print(f"[WARN] No rows for {filename}")
        return

    output_path = OUTPUT_DIR / filename
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows -> {output_path}")


def slice_by_family(rows: list[dict], family_name: str) -> list[dict]:
    return [r for r in rows if r["family"] == family_name]


def slice_by_tags(rows: list[dict], tag_list: list[str]) -> list[dict]:
    tag_set = set(tag_list)
    return [r for r in rows if r["primary_tag"] in tag_set]


def main() -> None:
    rows = load_rows()

    save_rows(slice_by_family(rows, "Creativity"), "labels_creativity.csv")
    save_rows(slice_by_family(rows, "Athletics"), "labels_athletics.csv")
    save_rows(slice_by_family(rows, "Smarts"), "labels_smarts.csv")
    save_rows(slice_by_family(rows, "Sociability"), "labels_sociability.csv")
    save_rows(slice_by_family(rows, "Vibes"), "labels_vibes.csv")

    save_rows(
        slice_by_tags(rows, ["Drawing", "Painting", "Photography", "Music"]),
        "labels_creativity_cluster.csv",
    )

    save_rows(
        slice_by_tags(rows, ["Basketball", "Running", "Workout", "Rock Climbing"]),
        "labels_athletics_cluster.csv",
    )

    save_rows(
        slice_by_tags(rows, ["Reading", "Study", "Chess"]),
        "labels_smarts_cluster.csv",
    )

    save_rows(
        slice_by_tags(rows, ["Travel", "Nature"]),
        "labels_vibes_cluster.csv",
    )


if __name__ == "__main__":
    main()