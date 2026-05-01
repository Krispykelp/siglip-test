import json
import sys
from pathlib import Path
from analyzer.engine import run_analysis

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "benchmark_debug"
DEFAULT_RESULTS_JSON = ROOT / "benchmark_results.json"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_safe_filename(image_path: str) -> str:
    return image_path.replace("/", "__").replace("\\", "__").replace(".", "_") + ".json"


def print_summary(result: dict) -> None:
    print("\n=== IMAGE ===")
    print(result["image_path"])

    family_stage = result.get("family_stage", {})
    tag_stage = result.get("tag_stage", {})

    family_results = family_stage.get("results", [])
    tag_results = tag_stage.get("results", [])

    print("\nTop Families:")
    if family_results:
        for family, score, _prompt in family_results[:5]:
            print(f"  {family}: {score:.3f}")
    else:
        print("  None")

    print("\nTop Tags:")
    if tag_results:
        for tag, score, _prompt in tag_results[:5]:
            print(f"  {tag}: {score:.3f}")
    else:
        print("  None")

    print("\nDecision:")
    print("  Family:", family_stage.get("decision_summary", {}).get("winner"))
    print("  Tag:", tag_stage.get("decision_summary", {}).get("winner"))


def normalize_image_path(image_path: str) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str((ROOT / image_path).resolve())


def inspect_image(image_path: str, output_dir: Path) -> None:
    resolved_image_path = normalize_image_path(image_path)
    result = run_analysis(image_path=resolved_image_path, claimed_tags=[])

    original_display_path = image_path
    result["image_path"] = original_display_path

    output_path = output_dir / make_safe_filename(original_display_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print_summary(result)
    print(f"\nSaved full JSON to: {output_path}")


def load_paths_from_txt(txt_path: Path) -> list[str]:
    with txt_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line and not line.startswith("#")]


def load_failures_from_results(
    results_json: Path,
    limit: int = 5,
    family: str | None = None,
    true_tag: str | None = None,
) -> list[str]:
    with results_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("results", [])
    failures = []

    for row in rows:
        if row.get("top1_correct"):
            continue
        if family and row.get("true_family") != family:
            continue
        if true_tag and row.get("true_tag") != true_tag:
            continue
        failures.append(row["image_path"])

    return failures[:limit]


def print_usage() -> None:
    print(
        "\nUsage:\n"
        "  python inspect_cases.py <image_path>\n"
        "  python inspect_cases.py <image_path1> <image_path2> ...\n"
        "  python inspect_cases.py --file benchmark_debug/paths.txt\n"
        "  python inspect_cases.py --from-results\n"
        "  python inspect_cases.py --from-results --family Creativity\n"
        "  python inspect_cases.py --from-results --tag Drawing\n"
        "  python inspect_cases.py --from-results --family Creativity --limit 8\n"
    )


def main() -> None:
    ensure_output_dir(DEFAULT_OUTPUT_DIR)

    args = sys.argv[1:]
    if not args:
        print_usage()
        return

    image_paths: list[str] = []

    if args[0] == "--file":
        if len(args) < 2:
            raise ValueError("Expected a text file path after --file")
        txt_path = Path(args[1])
        if not txt_path.is_absolute():
            txt_path = ROOT / txt_path
        if not txt_path.exists():
            raise FileNotFoundError(f"Text file not found: {txt_path}")
        image_paths = load_paths_from_txt(txt_path)

    elif args[0] == "--from-results":
        family = None
        true_tag = None
        limit = 5

        i = 1
        while i < len(args):
            if args[i] == "--family":
                family = args[i + 1]
                i += 2
            elif args[i] == "--tag":
                true_tag = args[i + 1]
                i += 2
            elif args[i] == "--limit":
                limit = int(args[i + 1])
                i += 2
            else:
                raise ValueError(f"Unknown argument: {args[i]}")

        image_paths = load_failures_from_results(
            results_json=DEFAULT_RESULTS_JSON,
            limit=limit,
            family=family,
            true_tag=true_tag,
        )

    else:
        image_paths = args

    if not image_paths:
        print("No image paths selected.")
        return

    print(f"Inspecting {len(image_paths)} image(s)...")
    for image_path in image_paths:
        inspect_image(image_path, DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()