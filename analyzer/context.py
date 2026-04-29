def infer_context_signals(detection_counts):
    person_count = detection_counts.get("person", 0)

    sports_items = sum(detection_counts.get(k, 0) for k in [
        "sports ball", "bicycle", "surfboard", "skateboard", "tennis racket",
        "frisbee", "baseball bat", "baseball glove", "skis", "snowboard", "kite"
    ])

    study_items = sum(detection_counts.get(k, 0) for k in [
        "laptop", "book", "backpack"
    ])

    social_scene_items = sum(detection_counts.get(k, 0) for k in [
        "dining table", "cake", "cup", "chair", "bench"
    ])

    travel_items = sum(detection_counts.get(k, 0) for k in [
        "car", "bus", "train", "airplane", "boat"
    ])

    pet_items = detection_counts.get("dog", 0) + detection_counts.get("cat", 0)
    phone_items = detection_counts.get("cell phone", 0)
    laptop_count = detection_counts.get("laptop", 0)
    book_count = detection_counts.get("book", 0)

    return {
        "person_count": person_count,
        "solo_person": person_count == 1,
        "pair_scene": person_count == 2,
        "group_scene": person_count >= 3,
        "large_group_scene": person_count >= 4,
        "sports_scene": sports_items > 0,
        "study_scene": study_items > 0,
        "social_scene_props": social_scene_items > 0,
        "pet_scene": pet_items > 0,
        "travel_scene": travel_items > 0,
        "phone_scene": phone_items > 0,
        "structured_social_scene": person_count >= 2 and social_scene_items > 0,
        "solo_study_scene": person_count == 1 and study_items > 0,
        "group_study_scene": person_count >= 2 and study_items > 0,
        "group_sports_scene": person_count >= 2 and sports_items > 0,
        "solo_vibes_scene": (
            (person_count <= 1 and pet_items > 0) or
            (person_count <= 1 and travel_items > 0) or
            (person_count <= 1 and phone_items > 0)
        ),
        "laptop_count": laptop_count,
        "book_count": book_count,
        "book_dominant_scene": book_count > laptop_count and book_count > 0,
        "laptop_heavy_scene": laptop_count >= 2 and book_count <= 1,
        "hybrid_laptop_book_scene": laptop_count >= 1 and book_count >= 1,
    }