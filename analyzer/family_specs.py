FAMILY_SPECS = {
    "Athletics": {
        "prompts": [
            "sports",
            "fitness activity",
            "athletic activity",
            "physical exercise",
            "outdoor sport",
            "competitive sport",
            "training session"
        ],
        "detector_weights": {
            "sports ball": 8.0,
            "bicycle": 7.0,
            "surfboard": 7.0,
            "skateboard": 7.0,
            "tennis racket": 7.0,
            "frisbee": 6.0,
            "baseball bat": 7.0,
            "baseball glove": 7.0,
            "skis": 7.0,
            "snowboard": 7.0,
            "kite": 5.5,
        },
        "modules": ["athletics_family"]
    },
    "Smarts": {
        "prompts": [
            "studying",
            "learning",
            "academic activity",
            "school work",
            "intellectual activity",
            "education",
            "focused desk work"
        ],
        "detector_weights": {
            "laptop": 6.0,
            "book": 3.5,
            "backpack": 1.0,
        },
        "modules": ["smarts_family"]
    },
    "Social": {
        "prompts": [
            "friends together",
            "group of people spending time together",
            "social gathering",
            "celebration with others",
            "family moment",
            "people interacting socially",
            "supportive social moment"
        ],
        "detector_weights": {
            "dining table": 3.0,
            "cake": 3.0,
            "cup": 1.0,
            "chair": 0.7,
            "bench": 0.6,
        },
        "modules": ["social_family"]
    },
    "Vibes": {
        "prompts": [
            "leisure activity",
            "casual relaxing moment",
            "lifestyle atmosphere",
            "travel or scenery moment",
            "pet moment",
            "nature leisure",
            "chill entertainment activity"
        ],
        "detector_weights": {
            "dog": 3.5,
            "cat": 3.5,
            "car": 0.8,
            "bus": 0.8,
            "train": 0.8,
            "airplane": 1.4,
            "boat": 1.2,
        },
        "modules": ["vibes_family"]
    },
    "Creativity": {
        "prompts": [
            "creative activity",
            "artistic work",
            "music or art",
            "creative project",
            "visual creativity",
            "self expression"
        ],
        "detector_weights": {
            "cell phone": 0.9,
        },
        "modules": ["creativity_family"]
    }
}