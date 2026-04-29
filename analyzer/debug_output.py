from .config import TOP_K_CANONICAL


def print_detections(all_counts, trusted_counts, stage_time):
    print(f"\nYOLO detection time: {stage_time:.3f}s")

    print("\nAll detected objects:")
    if not all_counts:
        print("- None")
    else:
        for cls_name, count in all_counts.items():
            print(f"- {cls_name}: {count}")

    print("\nTrusted detector objects used for routing:")
    if not trusted_counts:
        print("- None")
    else:
        for cls_name, count in trusted_counts.items():
            print(f"- {cls_name}: {count}")


def print_family_stage(family_results, top_families, stage_time, family_gate_ok, family_gate_reason):
    print(f"\nStage 1 family inference time: {stage_time:.3f}s")
    print("\nTop stat families:")
    for family, score, best_prompt in family_results:
        marker = "  <selected>" if family in top_families else ""
        print(f"- {family}: {score:.6f}  (best prompt: {best_prompt}){marker}")

    print("\nStage 1 gate:")
    print(f"- pass: {family_gate_ok}")
    print(f"- reason: {family_gate_reason}")


def print_context_signals(ctx):
    print("\nContext signals:")
    for key, value in ctx.items():
        print(f"- {key}: {value}")


def print_family_evidence(traced_family_results, decision_summary):
    print("\nFamily evidence breakdown:")
    for row in traced_family_results:
        print(
            f"\n[{row['family']}] "
            f"base={row['base_score']:.6f} -> final={row['final_score']:.6f} "
            f"(best prompt: {row['best_prompt']})"
        )

        if not row["adjustments"]:
            print("  - no heuristic adjustments")
        else:
            for adj in row["adjustments"]:
                sign = "+" if adj["delta"] >= 0 else ""
                print(f"  - {adj['source']}: {sign}{adj['delta']:.3f}")

    print("\nFamily decision summary:")
    print(f"- winner: {decision_summary['winner']}")
    print(f"- runner_up: {decision_summary['runner_up']}")
    print(f"- margin: {decision_summary['margin']}")
    print(f"- forced_family: {decision_summary['forced_family']}")
    print(f"- reason: {decision_summary['reason']}")


def print_tag_stage(
    canonical_results,
    claimed_tags,
    direct_supported,
    family_supported,
    unsupported,
    primary_inferred,
    bonus_inferred,
    rewards,
    stage_time
):
    print(f"\nStage 2 tag inference time: {stage_time:.3f}s")

    print("\nTop canonical AI tags:")
    for tag, score, best_prompt in canonical_results[:TOP_K_CANONICAL]:
        print(f"- {tag}: {score:.6f}  (best prompt: {best_prompt})")

    print("\nClaimed tags:")
    if claimed_tags:
        for tag in claimed_tags:
            print(f"- {tag}")
    else:
        print("- None")

    print("\nDirectly supported claimed tags:")
    if direct_supported:
        for tag, score in direct_supported:
            print(f"- {tag}: {score:.6f}")
    else:
        print("- None")

    print("\nFamily-supported claimed tags:")
    if family_supported:
        for tag, score in family_supported:
            print(f"- {tag}: {score:.6f}")
    else:
        print("- None")

    print("\nUnsupported claimed tags:")
    if unsupported:
        for tag, score in unsupported:
            if score is None:
                print(f"- {tag}: not present in selected family pool")
            else:
                print(f"- {tag}: {score:.6f}")
    else:
        print("- None")

    print("\nPrimary inferred tag:")
    if primary_inferred is not None:
        tag, score, best_prompt = primary_inferred
        print(f"- {tag}: {score:.6f}  (best prompt: {best_prompt})")
    else:
        print("- None")

    print("\nBonus inferred tags:")
    if bonus_inferred:
        for tag, score, best_prompt in bonus_inferred:
            print(f"- {tag}: {score:.6f}  (best prompt: {best_prompt})")
    else:
        print("- None")

    print("\nResolved rewards (strongest per family):")
    if rewards:
        for r in rewards:
            print(
                f"- stat={r['stat']}, tag={r['tag']}, "
                f"type={r['reward_type']}, amount={r['amount']}, score={r['score']:.6f}"
            )
    else:
        print("- None")


def print_tag_evidence(traced_tag_results, tag_decision_summary, limit=TOP_K_CANONICAL):
    print("\nTag evidence breakdown:")
    shown = traced_tag_results[:limit]

    for row in shown:
        print(
            f"\n[{row['tag']}] family={row['family']} "
            f"base={row['base_score']:.6f} -> final={row['final_score']:.6f} "
            f"(best prompt: {row['best_prompt']})"
        )

        if not row["adjustments"]:
            print("  - no heuristic adjustments")
        else:
            for adj in row["adjustments"]:
                sign = "+" if adj["delta"] >= 0 else ""
                print(f"  - {adj['source']}: {sign}{adj['delta']:.3f}")

    print("\nTag decision summary:")
    print(f"- winner: {tag_decision_summary['winner']}")
    print(f"- runner_up: {tag_decision_summary['runner_up']}")
    print(f"- margin: {tag_decision_summary['margin']}")
    print(f"- reason: {tag_decision_summary['reason']}")


def print_no_strong_match(claimed_tags):
    print("\nFINAL STATUS: no_strong_match")

    print("\nClaimed tags:")
    if claimed_tags:
        for tag in claimed_tags:
            print(f"- {tag}")
    else:
        print("- None")

    print("\nDirectly supported claimed tags:")
    print("- None")

    print("\nFamily-supported claimed tags:")
    print("- None")

    print("\nUnsupported claimed tags:")
    print("- None")

    print("\nPrimary inferred tag:")
    print("- None")

    print("\nBonus inferred tags:")
    print("- None")

    print("\nResolved rewards (strongest per family):")
    print("- None")


def print_analysis_result(result):
    detections = result["detections"]
    family_stage = result["family_stage"]
    tag_stage = result["tag_stage"]
    support = result["support"]

    print_detections(
        detections["all"],
        detections["trusted"],
        detections["stage_time_sec"],
    )

    family_results = family_stage["results"]
    top_families = [family for family, _, _ in family_results[:1]]

    print_family_stage(
        family_results=family_results,
        top_families=top_families,
        stage_time=family_stage["stage_time_sec"],
        family_gate_ok=family_stage["gate_pass"],
        family_gate_reason=family_stage["gate_reason"],
    )
    print_context_signals(family_stage["context_signals"])
    print_family_evidence(
        family_stage["evidence"],
        family_stage["decision_summary"],
    )

    if not family_stage["gate_pass"]:
        print_no_strong_match(result["claimed_tags"])
        return

    print_tag_stage(
        canonical_results=tag_stage["results"],
        claimed_tags=result["claimed_tags"],
        direct_supported=support["direct_supported"],
        family_supported=support["family_supported"],
        unsupported=support["unsupported"],
        primary_inferred=support["primary_inferred"],
        bonus_inferred=support["bonus_inferred"],
        rewards=result["resolved_rewards"],
        stage_time=tag_stage["stage_time_sec"],
    )
    print_tag_evidence(
        traced_tag_results=tag_stage["evidence"],
        tag_decision_summary=tag_stage["decision_summary"],
        limit=TOP_K_CANONICAL,
    )