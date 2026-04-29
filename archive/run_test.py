import json
from pathlib import Path
from pprint import pprint

from analyzer.engine import run_analysis
from analyzer.debug_output import print_analysis_result
from analyzer.schemas import make_compact_analysis_result
from analyzer.reward_pipeline import build_reward_summary_from_analysis


IMAGE_PATH = "test.jpg"
CLAIMED_TAGS = ["Basketball", "Workout"]

ANALYSIS_ID = "local-test-001"
POST_ID = None

PRINT_DEBUG_TO_TERMINAL = True
PRINT_REWARD_SUMMARY_TO_TERMINAL = True

WRITE_JSON = True
WRITE_DEBUG_JSON = True
WRITE_REWARD_JSON = True

OUTPUT_DIR = Path("outputs")
COMPACT_JSON_PATH = OUTPUT_DIR / "analysis_result.json"
DEBUG_JSON_PATH = OUTPUT_DIR / "analysis_result_debug.json"
REWARD_JSON_PATH = OUTPUT_DIR / "reward_result.json"


def write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    result = run_analysis(
        image_path=IMAGE_PATH,
        claimed_tags=CLAIMED_TAGS,
    )

    if PRINT_DEBUG_TO_TERMINAL:
        print_analysis_result(result)

    if WRITE_JSON:
        compact_result = make_compact_analysis_result(
            result,
            analysis_id=ANALYSIS_ID,
            post_id=POST_ID,
        )
        write_json_file(COMPACT_JSON_PATH, compact_result)
        print(f"\nWrote compact JSON to: {COMPACT_JSON_PATH.resolve()}")

    if WRITE_DEBUG_JSON:
        write_json_file(DEBUG_JSON_PATH, result)
        print(f"Wrote full debug JSON to: {DEBUG_JSON_PATH.resolve()}")

    reward_summary = build_reward_summary_from_analysis(result)

    if PRINT_REWARD_SUMMARY_TO_TERMINAL:
        print("\n=== Reward Summary ===")
        pprint(reward_summary, sort_dicts=False)

    if WRITE_REWARD_JSON:
        write_json_file(REWARD_JSON_PATH, reward_summary)
        print(f"Wrote reward JSON to: {REWARD_JSON_PATH.resolve()}")


if __name__ == "__main__":
    main()