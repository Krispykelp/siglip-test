from pathlib import Path
import csv

ROOT = Path("benchmark_dataset")
OUTPUT = Path("labels.csv")

CANONICAL_MAP = {
    "archery": ("Archery", "Athletics", True),
    "art": ("Art", "Creativity", True),
    "badminton": ("Badminton", "Athletics", True),
    "baking": ("Baking", "Creativity", True),
    "baseball": ("Baseball", "Athletics", True),
    "basketball": ("Basketball", "Athletics", True),
    "beach": ("Beach", "Vibes", True),
    "board_games": ("Board Games", "Smarts", True),
    "boxing": ("Boxing", "Athletics", True),
    "camping": ("Camping", "Vibes", True),
    "cardio": ("Cardio", "Athletics", True),
    "chess": ("Chess", "Smarts", True),
    "climbing": ("Climbing", "Athletics", True),
    "coding": ("Coding", "Smarts", True),
    "cooking": ("Cooking", "Creativity", True),
    "cycling": ("Cycling", "Athletics", True),
    "dance": ("Dance", "Creativity", True),
    "drawing": ("Drawing", "Creativity", True),
    "exam": ("Exam", "Smarts", True),
    "family": ("Family", "Social", True),
    "fashion": ("Fashion", "Creativity", True),
    "fishing": ("Fishing", "Vibes", True),
    "football": ("Football", "Athletics", True),
    "friends": ("Friends", "Social", True),
    "gaming": ("Gaming", "Vibes", True),
    "gardening": ("Gardening", "Vibes", True),
    "golf": ("Golf", "Athletics", True),
    "gym": ("Gym", "Athletics", True),
    "hangout": ("Hangout", "Social", True),
    "helping": ("Helping", "Social", True),
    "hiking": ("Hiking", "Athletics", True),
    "homework": ("Homework", "Smarts", True),
    "kayaking": ("Kayaking", "Athletics", True),
    "math": ("Math", "Smarts", True),
    "meditation": ("Meditation", "Vibes", True),
    "music": ("Music", "Creativity", True),
    "nature": ("Nature", "Vibes", True),
    "painting": ("Painting", "Creativity", True),
    "party": ("Party", "Social", True),
    "pets": ("Pets", "Vibes", True),
    "photography": ("Photography", "Creativity", True),
    "pickleball": ("Pickleball", "Athletics", True),
    "project": ("Project", "Smarts", True),
    "reading": ("Reading", "Smarts", True),
    "research": ("Research", "Smarts", True),
    "rock_climbing": ("Rock Climbing", "Athletics", True),
    "running": ("Running", "Athletics", True),
    "school": ("School", "Smarts", True),
    "science": ("Science", "Smarts", True),
    "selfie": ("Selfie", "Vibes", True),
    "skateboarding": ("Skateboarding", "Athletics", True),
    "skiing": ("Skiing", "Athletics", True),
    "soccer": ("Soccer", "Athletics", True),
    "study": ("Study", "Smarts", True),
    "support": ("Support", "Social", True),
    "surfing": ("Surfing", "Athletics", True),
    "swimming": ("Swimming", "Athletics", True),
    "table_tennis": ("Table Tennis", "Athletics", True),
    "tennis": ("Tennis", "Athletics", True),
    "travel": ("Travel", "Vibes", True),
    "volunteer": ("Volunteer", "Social", True),
    "volleyball": ("Volleyball", "Athletics", True),
    "walking": ("Walking", "Vibes", True),
    "weightlifting": ("Weightlifting", "Athletics", True),
    "workout": ("Workout", "Athletics", True),
    "writing": ("Writing", "Creativity", True),
    "yoga": ("Yoga", "Athletics", True),
    "ambiguous": ("None", "None", False),
    "hard_negatives": ("None", "None", False),
    "multi_activity": ("None", "None", False),
    "low_light": ("None", "None", False),
    "occluded": ("None", "None", False),
    "cluttered": ("None", "None", False),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    rows = []
    if not ROOT.exists():
        raise FileNotFoundError(f"Missing dataset folder: {ROOT.resolve()}")

    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir():
            continue

        key = folder.name.strip().lower()
        primary_tag, family, should_pass = CANONICAL_MAP.get(
            key,
            (folder.name.replace("_", " ").title(), "None", False),
        )

        for img in sorted(folder.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue

            rows.append({
                "image_path": str(img).replace("\\", "/"),
                "primary_tag": primary_tag,
                "secondary_tags": "",
                "family": family,
                "expected_should_pass": "1" if should_pass else "0",
                "notes": "",
            })

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "primary_tag",
                "secondary_tags",
                "family",
                "expected_should_pass",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} entries in {OUTPUT}")


if __name__ == "__main__":
    main()
