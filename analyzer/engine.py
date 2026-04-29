import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from .config import (
    TOP_FAMILIES_TO_KEEP,
    VLM_MODEL_NAME,
    YOLO_MODEL_NAME,
)
from .detection import (
    load_pil_image,
    load_yolo,
    run_yolo_detection,
    summarize_detections,
    summarize_trusted_detections,
    validate_image_for_pipeline,
)
from .family_specs import FAMILY_SPECS
from .schemas import make_analysis_result
from .scoring import (
    apply_family_fusion,
    apply_tag_fusion,
    build_verifier_summary,
    family_confidence_passes,
    resolve_rewards,
)
from .tag_specs import TAG_SPECS
from .vlm import (
    collapse_scores,
    flatten_nested_prompt_map,
    flatten_simple_prompt_map,
    load_vlm,
    print_device_info,
    run_vlm,
    sort_collapsed,
)


def run_analysis(image_path: str, claimed_tags: list[str]):
    validate_image_for_pipeline(image_path)

    device = print_device_info()
    processor, vlm = load_vlm(VLM_MODEL_NAME, device)
    detector = load_yolo(YOLO_MODEL_NAME)

    detections, detect_time = run_yolo_detection(detector, image_path)
    all_detection_counts = summarize_detections(detections)
    trusted_detection_counts, _ = summarize_trusted_detections(detections)

    family_prompt_map = {family: spec["prompts"] for family, spec in FAMILY_SPECS.items()}
    family_prompts, family_keys = flatten_simple_prompt_map(family_prompt_map)
    family_scores, family_time = run_vlm(
        processor=processor,
        model=vlm,
        device=device,
        image_path=image_path,
        all_prompts=family_prompts,
        load_pil_image_fn=load_pil_image,
    )

    collapsed_families = collapse_scores(
        keys=family_keys,
        scores=family_scores,
        prompts=family_prompts,
    )
    family_results = sort_collapsed(collapsed_families)
    family_results, traced_family_results, ctx, decision_summary = apply_family_fusion(
        family_results,
        trusted_detection_counts,
    )

    top_families = [family for family, _, _ in family_results[:TOP_FAMILIES_TO_KEEP]]
    family_gate_ok, family_gate_reason = family_confidence_passes(
        family_results,
        decision_summary["forced_family"],
    )

    canonical_results = []
    traced_tag_results = []
    tag_decision_summary = {
        "winner": None,
        "runner_up": None,
        "margin": None,
        "reason": "tag_stage_not_run"
    }
    tag_time = 0.0
    direct_supported = []
    family_supported = []
    unsupported = []
    primary_inferred = None
    bonus_inferred = []
    rewards = []

    if family_gate_ok:
        selected_tag_map = {}
        for tag, spec in TAG_SPECS.items():
            if spec["family"] in top_families:
                selected_tag_map[tag] = spec["prompts"]

        tag_prompts, tag_keys = flatten_nested_prompt_map(selected_tag_map)
        tag_scores, tag_time = run_vlm(
            processor=processor,
            model=vlm,
            device=device,
            image_path=image_path,
            all_prompts=tag_prompts,
            load_pil_image_fn=load_pil_image,
        )

        collapsed_tags = collapse_scores(
            keys=tag_keys,
            scores=tag_scores,
            prompts=tag_prompts,
        )
        canonical_results = sort_collapsed(collapsed_tags)
        canonical_results, traced_tag_results, tag_decision_summary = apply_tag_fusion(
            canonical_results,
            trusted_detection_counts,
        )

        top_family = top_families[0]
        direct_supported, family_supported, unsupported, primary_inferred, bonus_inferred = build_verifier_summary(
            canonical_results=canonical_results,
            claimed_tags=claimed_tags,
            top_family=top_family,
        )

        rewards = resolve_rewards(
            direct_supported=direct_supported,
            family_supported=family_supported,
            primary_inferred=primary_inferred,
            bonus_inferred=bonus_inferred,
        )

    return make_analysis_result(
        image_path=image_path,
        claimed_tags=claimed_tags,
        detections_all=all_detection_counts,
        detections_trusted=trusted_detection_counts,
        detection_time=detect_time,
        family_results=family_results,
        traced_family_results=traced_family_results,
        family_time=family_time,
        family_gate_ok=family_gate_ok,
        family_gate_reason=family_gate_reason,
        context_signals=ctx,
        family_decision_summary=decision_summary,
        canonical_results=canonical_results,
        traced_tag_results=traced_tag_results,
        tag_decision_summary=tag_decision_summary,
        tag_time=tag_time,
        direct_supported=direct_supported,
        family_supported=family_supported,
        unsupported=unsupported,
        primary_inferred=primary_inferred,
        bonus_inferred=bonus_inferred,
        rewards=rewards,
    )