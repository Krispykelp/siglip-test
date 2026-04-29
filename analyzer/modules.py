from .config import ACTIVITY_BIAS


def add_adjustment(adjustments, source, delta):
    adjustments.append({"source": source, "delta": delta})


def apply_weight_map(adjusted, adjustments, weight_map, counts, label_prefix):
    for key, weight in weight_map.items():
        count = counts.get(key, 0)
        if count > 0:
            delta = weight * count
            adjusted += delta
            add_adjustment(adjustments, f"{label_prefix}:{key}x{count}", delta)
    return adjusted


def apply_context_weight_map(adjusted, adjustments, weight_map, ctx, label_prefix):
    for key, weight in weight_map.items():
        if ctx.get(key, False):
            adjusted += weight
            add_adjustment(adjustments, f"{label_prefix}:{key}", weight)
    return adjusted


def apply_module(module_name, adjusted, adjustments, detection_counts, ctx, entity_name, entity_type):
    if module_name == "activity_bias":
        adjusted += ACTIVITY_BIAS
        add_adjustment(adjustments, "global:activity_bias", ACTIVITY_BIAS)

    elif module_name == "athletics_family":
        if ctx["group_sports_scene"]:
            adjusted += 2.5
            add_adjustment(adjustments, "ctx:group_sports_scene", 2.5)
        if ctx["sports_scene"] and ctx["solo_person"]:
            adjusted += 1.0
            add_adjustment(adjustments, "ctx:solo_sports_scene", 1.0)
        if ctx["study_scene"]:
            adjusted -= 1.5
            add_adjustment(adjustments, "penalty:study_scene", -1.5)

    elif module_name == "smarts_family":
        if detection_counts.get("laptop", 0) > 0 and detection_counts.get("person", 0) > 0:
            adjusted += 5.0
            add_adjustment(adjustments, "ctx:laptop+person", 5.0)
        if detection_counts.get("book", 0) > 0 and detection_counts.get("person", 0) > 0:
            adjusted += 2.5
            add_adjustment(adjustments, "ctx:book+person", 2.5)
        if ctx["solo_study_scene"]:
            adjusted += 2.0
            add_adjustment(adjustments, "ctx:solo_study_scene", 2.0)
        if ctx["group_study_scene"]:
            adjusted += 1.8
            add_adjustment(adjustments, "ctx:group_study_scene", 1.8)
        if ctx["sports_scene"]:
            adjusted -= 2.0
            add_adjustment(adjustments, "penalty:sports_scene", -2.0)

    elif module_name == "social_family":
        if ctx["pair_scene"] and not ctx["sports_scene"] and not ctx["study_scene"]:
            adjusted += 2.2
            add_adjustment(adjustments, "ctx:pair_scene", 2.2)
        if ctx["group_scene"] and not ctx["sports_scene"] and not ctx["study_scene"]:
            adjusted += 4.2
            add_adjustment(adjustments, "ctx:group_scene", 4.2)
        if ctx["structured_social_scene"]:
            adjusted += 3.5
            add_adjustment(adjustments, "ctx:structured_social_scene", 3.5)
        if detection_counts.get("cake", 0) > 0 and detection_counts.get("person", 0) >= 2:
            adjusted += 3.0
            add_adjustment(adjustments, "ctx:cake+group", 3.0)
        if ctx["study_scene"] and ctx["group_scene"]:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:group_study_social_bonus", 1.2)
        if ctx["sports_scene"] and ctx["group_scene"]:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:team_sports_social_bonus", 0.8)
        if ctx["solo_person"]:
            adjusted -= 1.5
            add_adjustment(adjustments, "penalty:solo_person", -1.5)

    elif module_name == "vibes_family":
        if ctx["pet_scene"]:
            adjusted += 3.0
            add_adjustment(adjustments, "ctx:pet_scene", 3.0)
        if ctx["travel_scene"]:
            adjusted += 2.2
            add_adjustment(adjustments, "ctx:travel_scene", 2.2)
        if ctx["solo_vibes_scene"]:
            adjusted += 1.6
            add_adjustment(adjustments, "ctx:solo_vibes_scene", 1.6)
        if ctx["group_scene"] and not ctx["pet_scene"] and not ctx["travel_scene"]:
            adjusted -= 2.5
            add_adjustment(adjustments, "penalty:group_scene_not_vibes", -2.5)
        if ctx["sports_scene"]:
            adjusted -= 2.0
            add_adjustment(adjustments, "penalty:sports_scene", -2.0)
        if ctx["study_scene"]:
            adjusted -= 1.5
            add_adjustment(adjustments, "penalty:study_scene", -1.5)
        if ctx["structured_social_scene"]:
            adjusted -= 2.2
            add_adjustment(adjustments, "penalty:structured_social_scene", -2.2)

    elif module_name == "creativity_family":
        if detection_counts.get("cell phone", 0) > 0:
            adjusted += 0.5
            add_adjustment(adjustments, "ctx:cell_phone_present", 0.5)
        if detection_counts.get("person", 0) == 1 and detection_counts.get("cell phone", 0) > 0:
            adjusted += 0.5
            add_adjustment(adjustments, "ctx:solo_phone_scene", 0.5)

    elif module_name == "screen_work":
        if detection_counts.get("laptop", 0) > 0 and detection_counts.get("person", 0) > 0:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:laptop+person_for_study_or_coding", 1.2)
        if detection_counts.get("laptop", 0) > 0:
            adjusted += 2.4
            add_adjustment(adjustments, "ctx:laptop_for_coding", 2.4)
        if ctx["laptop_heavy_scene"]:
            adjusted += 2.2
            add_adjustment(adjustments, "ctx:multiple_laptops_for_coding", 2.2)
        if ctx["group_study_scene"] and ctx["laptop_count"] >= 2:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:group_laptop_scene_for_coding", 1.2)
        if ctx["book_dominant_scene"]:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:book_dominant_scene", -1.0)

    elif module_name == "study_general":
        if detection_counts.get("laptop", 0) > 0 and detection_counts.get("person", 0) > 0:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:laptop+person_for_study_or_coding", 1.2)
        if detection_counts.get("book", 0) > 0:
            adjusted += 2.0
            add_adjustment(adjustments, "ctx:book_for_study", 2.0)
        if ctx["laptop_heavy_scene"]:
            adjusted -= 0.8
            add_adjustment(adjustments, "penalty:laptop_heavy_scene", -0.8)

    elif module_name == "reading_like":
        if detection_counts.get("book", 0) > 0:
            adjusted += 2.5
            add_adjustment(adjustments, "ctx:book_for_reading", 2.5)
        if ctx["laptop_count"] >= 2:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:multi_laptop_scene", -1.0)

    elif module_name == "research_like":
        if ctx["hybrid_laptop_book_scene"]:
            adjusted += 1.6
            add_adjustment(adjustments, "ctx:laptop+book_for_research", 1.6)

    elif module_name == "project_like":
        if detection_counts.get("laptop", 0) > 0:
            adjusted += 1.0
            add_adjustment(adjustments, "ctx:laptop_for_project", 1.0)
        if ctx["group_study_scene"] and ctx["laptop_count"] >= 2:
            adjusted += 1.4
            add_adjustment(adjustments, "ctx:group_laptop_project_scene", 1.4)

    elif module_name == "homework_like":
        if ctx["study_scene"] and detection_counts.get("person", 0) >= 1:
            adjusted += 1.3
            add_adjustment(adjustments, "ctx:study_scene_for_homework", 1.3)

    elif module_name == "school_like":
        if detection_counts.get("backpack", 0) > 0:
            adjusted += 1.5
            add_adjustment(adjustments, "ctx:backpack_for_school", 1.5)

    elif module_name == "group_social":
        if ctx["group_scene"] and not ctx["sports_scene"] and not ctx["study_scene"]:
            adjusted += 2.4
            add_adjustment(adjustments, "ctx:group_scene_for_friends", 2.4)
        if ctx["solo_person"] and not ctx["structured_social_scene"]:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:solo_person", -1.0)

    elif module_name == "hangout_like":
        if ctx["pair_scene"] and not ctx["sports_scene"] and not ctx["study_scene"]:
            adjusted += 1.8
            add_adjustment(adjustments, "ctx:pair_scene_for_hangout", 1.8)
        if ctx["structured_social_scene"]:
            adjusted += 2.5
            add_adjustment(adjustments, "ctx:structured_social_scene_for_hangout", 2.5)
        if ctx["solo_person"] and not ctx["structured_social_scene"]:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:solo_person", -1.0)

    elif module_name == "family_like":
        if ctx["group_scene"] and detection_counts.get("dining table", 0) > 0:
            adjusted += 1.8
            add_adjustment(adjustments, "ctx:group_table_for_family", 1.8)
        if ctx["solo_person"] and not ctx["structured_social_scene"]:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:solo_person", -1.0)

    elif module_name == "party_like":
        if detection_counts.get("cake", 0) > 0 and detection_counts.get("person", 0) >= 2:
            adjusted += 3.0
            add_adjustment(adjustments, "ctx:cake+group_for_party", 3.0)
        if ctx["solo_person"] and not ctx["structured_social_scene"]:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:solo_person", -1.0)

    elif module_name == "supportive_social":
        if detection_counts.get("person", 0) >= 2:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:multi_person_for_supportive_social", 0.8)
        if ctx["solo_person"] and not ctx["structured_social_scene"]:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:solo_person", -1.0)

    elif module_name == "pet_leisure":
        if detection_counts.get("dog", 0) > 0 or detection_counts.get("cat", 0) > 0:
            adjusted += 2.5
            add_adjustment(adjustments, "ctx:pet_detector_for_pets", 2.5)
        if ctx["group_scene"] and not ctx["pet_scene"] and not ctx["travel_scene"]:
            adjusted -= 1.8
            add_adjustment(adjustments, "penalty:group_scene_not_vibes", -1.8)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)
        if ctx["sports_scene"]:
            adjusted -= 1.5
            add_adjustment(adjustments, "penalty:sports_scene", -1.5)

    elif module_name == "travel_like":
        if ctx["travel_scene"]:
            adjusted += 2.4
            add_adjustment(adjustments, "ctx:vehicle_or_trip_detector_for_travel", 2.4)
        if ctx["group_scene"] and not ctx["pet_scene"] and not ctx["travel_scene"]:
            adjusted -= 1.8
            add_adjustment(adjustments, "penalty:group_scene_not_vibes", -1.8)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)
        if ctx["sports_scene"]:
            adjusted -= 1.5
            add_adjustment(adjustments, "penalty:sports_scene", -1.5)

    elif module_name == "selfie_like":
        if detection_counts.get("cell phone", 0) > 0 and detection_counts.get("person", 0) == 1:
            adjusted += 2.0
            add_adjustment(adjustments, "ctx:solo_phone_scene_for_selfie", 2.0)
        if ctx["group_scene"] and not ctx["pet_scene"] and not ctx["travel_scene"]:
            adjusted -= 1.8
            add_adjustment(adjustments, "penalty:group_scene_not_vibes", -1.8)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)
        if ctx["sports_scene"]:
            adjusted -= 1.5
            add_adjustment(adjustments, "penalty:sports_scene", -1.5)

    elif module_name == "walking_like":
        if detection_counts.get("dog", 0) > 0:
            adjusted += 1.5
            add_adjustment(adjustments, "ctx:dog_for_walking", 1.5)
        if ctx["group_scene"] and not ctx["pet_scene"] and not ctx["travel_scene"]:
            adjusted -= 1.8
            add_adjustment(adjustments, "penalty:group_scene_not_vibes", -1.8)

    elif module_name == "beach_like":
        if (
            detection_counts.get("surfboard", 0) > 0
            or detection_counts.get("kite", 0) > 0
            or detection_counts.get("boat", 0) > 0
        ):
            adjusted += 1.8
            add_adjustment(adjustments, "ctx:beach_related_detector", 1.8)
        if ctx["group_scene"] and not ctx["pet_scene"] and not ctx["travel_scene"]:
            adjusted -= 1.8
            add_adjustment(adjustments, "penalty:group_scene_not_vibes", -1.8)

    elif module_name == "screen_leisure":
        if ctx["group_scene"] and not ctx["pet_scene"] and not ctx["travel_scene"]:
            adjusted -= 1.8
            add_adjustment(adjustments, "penalty:group_scene_not_vibes", -1.8)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "photography_like":
        if detection_counts.get("cell phone", 0) > 0:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:cell_phone_for_photography", 1.2)

    elif module_name == "fashion_like":
        if detection_counts.get("cell phone", 0) > 0 and detection_counts.get("person", 0) >= 1:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:person+phone_for_fashion", 0.8)

    elif module_name == "ball_sport_like":
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "workout_like":
        if detection_counts.get("person", 0) >= 2:
            adjusted += 1.0
            add_adjustment(adjustments, "ctx:workout_multi_person", 1.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "cycling_like":
        if detection_counts.get("bicycle", 0) > 0:
            adjusted += 2.0
            add_adjustment(adjustments, "ctx:bicycle_for_cycling", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "surf_like":
        if detection_counts.get("surfboard", 0) > 0:
            adjusted += 2.0
            add_adjustment(adjustments, "ctx:surfboard_for_surfing", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "skate_like":
        if detection_counts.get("skateboard", 0) > 0:
            adjusted += 2.0
            add_adjustment(adjustments, "ctx:skateboard_for_skateboarding", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "racket_sport_like":
        if detection_counts.get("tennis racket", 0) > 0:
            adjusted += 2.0
            add_adjustment(adjustments, "ctx:racket_for_tennis", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "baseball_like":
        if detection_counts.get("baseball bat", 0) > 0 or detection_counts.get("baseball glove", 0) > 0:
            adjusted += 2.0
            add_adjustment(adjustments, "ctx:baseball_gear", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "snow_sport_like":
        if detection_counts.get("skis", 0) > 0 or detection_counts.get("snowboard", 0) > 0:
            adjusted += 2.0
            add_adjustment(adjustments, "ctx:snow_sport_gear", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    else:
        add_adjustment(adjustments, f"warning:unknown_module:{module_name}", 0.0)

    return adjusted