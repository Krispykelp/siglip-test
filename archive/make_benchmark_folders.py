from pathlib import Path
import re

ROOT = Path("benchmark_dataset")

tags = [
    "Archery",
    "Art",
    "Badminton",
    "Baking",
    "Baseball",
    "Basketball",
    "Beach",
    "Board Games",
    "Boxing",
    "Camping",
    "Cardio",
    "Chess",
    "Climbing",
    "Coding",
    "Cooking",
    "Cycling",
    "Dance",
    "Drawing",
    "Exam",
    "Family",
    "Fashion",
    "Fishing",
    "Football",
    "Friends",
    "Gaming",
    "Gardening",
    "Golf",
    "Gym",
    "Hangout",
    "Helping",
    "Hiking",
    "Homework",
    "Kayaking",
    "Math",
    "Meditation",
    "Music",
    "Nature",
    "Painting",
    "Party",
    "Pets",
    "Photography",
    "Pickleball",
    "Project",
    "Reading",
    "Research",
    "Rock Climbing",
    "Running",
    "Science",
    "School",
    "Selfie",
    "Skateboarding",
    "Skiing",
    "Soccer",
    "Study",
    "Support",
    "Surfing",
    "Swimming",
    "Table Tennis",
    "Tennis",
    "Travel",
    "Volunteer",
    "Volleyball",
    "Walking",
    "Weightlifting",
    "Workout",
    "Writing",
    "Yoga",
]

def slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s-]+", "_", name)
    return name

def main():
    ROOT.mkdir(exist_ok=True)

    for tag in tags:
        folder = ROOT / slugify(tag)
        folder.mkdir(parents=True, exist_ok=True)

    # optional extra buckets
    extras = [
        "hard_negatives",
        "ambiguous",
        "multi_activity",
        "low_light",
        "occluded",
        "cluttered",
    ]
    for name in extras:
        (ROOT / name).mkdir(parents=True, exist_ok=True)

    print(f"Created {len(tags) + len(extras)} folders under '{ROOT}'")

if __name__ == "__main__":
    main()