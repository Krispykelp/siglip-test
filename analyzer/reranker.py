from typing import Any


def _strong_sports_count(detection_counts: dict[str, int]) -> int:
    return (
        detection_counts.get("sports ball", 0)
        + detection_counts.get("bicycle", 0)
        + detection_counts.get("surfboard", 0)
        + detection_counts.get("skateboard", 0)
        + detection_counts.get("tennis racket", 0)
        + detection_counts.get("frisbee", 0)
        + detection_counts.get("baseball bat", 0)
        + detection_counts.get("baseball glove", 0)
        + detection_counts.get("skis", 0)
        + detection_counts.get("snowboard", 0)
    )


def rerank_tag_candidates(
    tag_results: list[tuple[str, float, str]],
    detection_counts: dict[str, int],
    ctx: dict[str, Any],
    family_results: list[tuple[str, float, str]] | None = None,
) -> tuple[list[tuple[str, float, str]], dict[str, Any]]:
    if not tag_results:
        return [], {
            "winner": None,
            "runner_up": None,
            "margin": None,
            "reason": "rerank_skipped_no_candidates",
            "adjustments": [],
        }

    table: dict[str, dict[str, Any]] = {}
    for tag, score, prompt in tag_results:
        table[tag] = {
            "score": float(score),
            "prompt": prompt,
        }

    adjustments: list[dict[str, Any]] = []

    def bump(tag: str, delta: float, reason: str) -> None:
        if tag not in table:
            return
        table[tag]["score"] += delta
        adjustments.append(
            {
                "tag": tag,
                "delta": delta,
                "reason": reason,
            }
        )

    person_count = int(ctx.get("person_count", 0))
    phone_count = int(detection_counts.get("cell phone", 0))
    book_count = int(ctx.get("book_count", 0))
    laptop_count = int(ctx.get("laptop_count", 0))
    sports_count = _strong_sports_count(detection_counts)

    no_people = person_count == 0
    no_phone = phone_count == 0
    travel_scene = bool(ctx.get("travel_scene", False))
    study_scene = bool(ctx.get("study_scene", False))
    social_scene_props = bool(ctx.get("social_scene_props", False))
    structured_social_scene = bool(ctx.get("structured_social_scene", False))
    solo_person = bool(ctx.get("solo_person", False))
    pair_scene = bool(ctx.get("pair_scene", False))

    open_scene = (
        not study_scene
        and sports_count == 0
        and not social_scene_props
        and not structured_social_scene
    )

    family_winner = family_results[0][0] if family_results else None

    # =========================
    # PHOTOGRAPHY DE-ATTRACTOR
    # =========================
    if "Photography" in table:
        if no_phone:
            bump("Photography", -4.8, "rerank:no_phone_for_photography")
        if no_people:
            bump("Photography", -1.2, "rerank:no_people_for_photography")
        if travel_scene:
            bump("Photography", -2.0, "rerank:travel_scene_for_photography")
        if sports_count > 0:
            bump("Photography", -2.8, "rerank:sports_scene_for_photography")
        if study_scene:
            bump("Photography", -1.8, "rerank:study_scene_for_photography")
        if book_count > 0 and no_phone:
            bump("Photography", -1.2, "rerank:book_scene_for_photography")
        if social_scene_props and no_phone:
            bump("Photography", -1.0, "rerank:social_props_for_photography")

    # =========================
    # OPEN-SCENE CREATIVE RESCUE
    # Narrowed to avoid overpromoting Music
    # =========================
    if open_scene and no_phone:
        if no_people:
            bump("Drawing", 4.0, "rerank:no_people_open_scene_for_drawing")
            bump("Painting", 3.4, "rerank:no_people_open_scene_for_painting")
            bump("Music", 1.0, "rerank:no_people_open_scene_for_music")
        else:
            bump("Painting", 2.2, "rerank:open_scene_for_painting")
            bump("Drawing", 1.8, "rerank:open_scene_for_drawing")
            bump("Music", 0.8, "rerank:open_scene_for_music")

    # =========================
    # VIBES RESCUE
    # Prefer Travel/Nature in open scenic scenes
    # =========================
    if open_scene and no_phone:
        if no_people:
            bump("Nature", 2.8, "rerank:no_people_open_scene_for_nature")
            bump("Travel", 2.0, "rerank:no_people_open_scene_for_travel")
        elif solo_person:
            bump("Travel", 2.2, "rerank:solo_open_scene_for_travel")
            bump("Nature", 1.0, "rerank:solo_open_scene_for_nature")
        elif pair_scene:
            bump("Travel", 1.2, "rerank:pair_open_scene_for_travel")

    if travel_scene:
        bump("Travel", 2.4, "rerank:travel_scene_for_travel")
        bump("Nature", 0.6, "rerank:travel_scene_for_nature")
        bump("Music", -1.8, "rerank:travel_scene_against_music")
        bump("Painting", -0.8, "rerank:travel_scene_against_painting")
        bump("Drawing", -0.8, "rerank:travel_scene_against_drawing")

    # =========================
    # SMARTS NARROWING
    # =========================
    if book_count == 0 and laptop_count == 0 and not study_scene:
        for tag in ["Study", "Research", "Project", "Homework", "School", "Exam", "Math", "Coding"]:
            if tag == "Research":
                bump(tag, -3.6, "rerank:no_study_context_for_research")
            else:
                bump(tag, -2.4, f"rerank:no_study_context_for_{tag.lower()}")

    if book_count > 0 and laptop_count == 0:
        bump("Reading", 2.0, "rerank:book_dominant_for_reading")
        bump("Study", -1.0, "rerank:book_dominant_against_study")
        bump("Research", -1.2, "rerank:book_dominant_against_research")

    if not travel_scene and sports_count == 0 and not social_scene_props:
        bump("Chess", 1.6, "rerank:quiet_scene_for_chess")
        bump("Music", -0.6, "rerank:quiet_scene_against_music")

    # =========================
    # ATHLETICS RESCUE
    # =========================
    if sports_count > 0:
        bump("Music", -2.2, "rerank:sports_scene_against_music")

        if detection_counts.get("sports ball", 0) > 0:
            bump("Basketball", 1.8, "rerank:sports_ball_for_basketball")
            bump("Soccer", 1.0, "rerank:sports_ball_for_soccer")
            bump("Football", 1.0, "rerank:sports_ball_for_football")
            bump("Volleyball", 1.0, "rerank:sports_ball_for_volleyball")
            bump("Baseball", 0.8, "rerank:sports_ball_for_baseball")

        if detection_counts.get("tennis racket", 0) > 0:
            bump("Tennis", 2.0, "rerank:racket_for_tennis")

        if detection_counts.get("bicycle", 0) > 0:
            bump("Cycling", 2.0, "rerank:bicycle_for_cycling")

        if detection_counts.get("surfboard", 0) > 0:
            bump("Surfing", 2.0, "rerank:surfboard_for_surfing")

        if detection_counts.get("skateboard", 0) > 0:
            bump("Skateboarding", 2.0, "rerank:skateboard_for_skateboarding")

        if detection_counts.get("skis", 0) > 0 or detection_counts.get("snowboard", 0) > 0:
            bump("Skiing", 2.0, "rerank:snow_gear_for_skiing")

        if detection_counts.get("baseball bat", 0) > 0 or detection_counts.get("baseball glove", 0) > 0:
            bump("Baseball", 2.0, "rerank:baseball_gear_for_baseball")

        # Open athletic fallback when sports evidence exists
        bump("Running", 0.8, "rerank:sports_scene_for_running")
        bump("Workout", 0.8, "rerank:sports_scene_for_workout")
        bump("Cardio", 0.6, "rerank:sports_scene_for_cardio")
        bump("Rock Climbing", 0.6, "rerank:sports_scene_for_rock_climbing")

    elif solo_person and open_scene and not travel_scene:
        bump("Running", 1.6, "rerank:solo_open_scene_for_running")
        bump("Workout", 1.2, "rerank:solo_open_scene_for_workout")
        bump("Cardio", 1.2, "rerank:solo_open_scene_for_cardio")
        bump("Hiking", 1.0, "rerank:solo_open_scene_for_hiking")
        bump("Rock Climbing", 1.0, "rerank:solo_open_scene_for_rock_climbing")
        bump("Music", -0.8, "rerank:solo_open_scene_against_music")

    # =========================
    # CREATIVITY FAMILY TIEBREAKER
    # Prefer Drawing/Painting over Music in generic creative scenes
    # =========================
    if family_winner == "Creativity" and open_scene and no_phone:
        bump("Drawing", 1.0, "rerank:family_creativity_bias_for_drawing")
        bump("Painting", 1.4, "rerank:family_creativity_bias_for_painting")
        bump("Music", -0.8, "rerank:family_creativity_bias_against_music")

    # =========================
    # FINAL NEAR-TIE PREFERENCES
    # If Music is only narrowly ahead, prefer the more scene-grounded tag
    # =========================
    def score_of(tag: str) -> float | None:
        row = table.get(tag)
        return None if row is None else float(row["score"])

    music_score = score_of("Music")

    if music_score is not None:
        for rival, delta, threshold, reason in [
            ("Travel", 0.8, 2.0, "rerank:near_tie_prefer_travel_over_music"),
            ("Nature", 0.8, 2.0, "rerank:near_tie_prefer_nature_over_music"),
            ("Painting", 0.8, 1.8, "rerank:near_tie_prefer_painting_over_music"),
            ("Drawing", 0.8, 1.8, "rerank:near_tie_prefer_drawing_over_music"),
            ("Running", 0.8, 1.8, "rerank:near_tie_prefer_running_over_music"),
            ("Workout", 0.8, 1.8, "rerank:near_tie_prefer_workout_over_music"),
            ("Chess", 0.8, 1.8, "rerank:near_tie_prefer_chess_over_music"),
        ]:
            rival_score = score_of(rival)
            if rival_score is None:
                continue
            if music_score >= rival_score and (music_score - rival_score) <= threshold:
                bump(rival, delta, reason)

    reranked = sorted(
        [(tag, row["score"], row["prompt"]) for tag, row in table.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    winner = reranked[0][0] if reranked else None
    runner_up = reranked[1][0] if len(reranked) > 1 else None
    margin = (reranked[0][1] - reranked[1][1]) if len(reranked) > 1 else None

    return reranked, {
        "winner": winner,
        "runner_up": runner_up,
        "margin": margin,
        "reason": "post_fusion_rerank",
        "adjustments": adjustments,
    }