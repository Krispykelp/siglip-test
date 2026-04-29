from .family_specs import FAMILY_SPECS

TAG_SPECS = {
    "Basketball": {
        "family": "Athletics",
        "prompts": [
            "Basketball",
            "basketball game",
            "outdoor basketball",
            "person dribbling a basketball"
        ],
        "detector_weights": {"sports ball": 8.0},
        "modules": ["activity_bias", "ball_sport_like"],
    },
    "Workout": {
        "family": "Athletics",
        "prompts": [
            "Workout",
            "gym workout",
            "exercise session",
            "fitness training"
        ],
        "modules": ["activity_bias", "workout_like"],
    },
    "Hiking": {
        "family": "Athletics",
        "prompts": [
            "Hiking",
            "outdoor hiking",
            "trail walking",
            "hiking outdoors"
        ],
        "modules": ["activity_bias"],
    },
    "Running": {
        "family": "Athletics",
        "prompts": [
            "Running",
            "outdoor running",
            "jogging",
            "running exercise"
        ],
        "modules": ["activity_bias"],
    },
    "Soccer": {
        "family": "Athletics",
        "prompts": [
            "Soccer",
            "soccer game",
            "playing soccer",
            "outdoor soccer"
        ],
        "detector_weights": {"sports ball": 5.0},
        "modules": ["activity_bias", "ball_sport_like"],
    },
    "Cycling": {
        "family": "Athletics",
        "prompts": [
            "Cycling",
            "riding a bicycle",
            "biking outdoors",
            "cycling activity"
        ],
        "detector_weights": {"bicycle": 8.0},
        "modules": ["activity_bias", "cycling_like"],
    },
    "Surfing": {
        "family": "Athletics",
        "prompts": [
            "Surfing",
            "riding a surfboard",
            "surfing on water",
            "surf activity"
        ],
        "detector_weights": {"surfboard": 8.0},
        "modules": ["activity_bias", "surf_like"],
    },
    "Skateboarding": {
        "family": "Athletics",
        "prompts": [
            "Skateboarding",
            "riding a skateboard",
            "skateboard trick",
            "skateboarding activity"
        ],
        "detector_weights": {"skateboard": 8.0},
        "modules": ["activity_bias", "skate_like"],
    },
    "Tennis": {
        "family": "Athletics",
        "prompts": [
            "Tennis",
            "playing tennis",
            "tennis racket sport",
            "tennis match"
        ],
        "detector_weights": {"tennis racket": 8.0},
        "modules": ["activity_bias", "racket_sport_like"],
    },
    "Baseball": {
        "family": "Athletics",
        "prompts": [
            "Baseball",
            "baseball game",
            "baseball bat sport",
            "playing baseball"
        ],
        "detector_weights": {
            "baseball bat": 8.0,
            "baseball glove": 7.0
        },
        "modules": ["activity_bias", "baseball_like"],
    },
    "Skiing": {
        "family": "Athletics",
        "prompts": [
            "Skiing",
            "skiing on snow",
            "ski activity",
            "winter skiing"
        ],
        "detector_weights": {
            "skis": 8.0,
            "snowboard": 4.5
        },
        "modules": ["activity_bias", "snow_sport_like"],
    },
    "Football": {
        "family": "Athletics",
        "prompts": [
            "Football",
            "football game",
            "playing football",
            "team football"
        ],
        "detector_weights": {"sports ball": 5.0},
        "modules": ["activity_bias", "ball_sport_like"],
    },
    "Volleyball": {
        "family": "Athletics",
        "prompts": [
            "Volleyball",
            "volleyball game",
            "playing volleyball",
            "beach volleyball"
        ],
        "detector_weights": {"sports ball": 5.0},
        "modules": ["activity_bias", "ball_sport_like"],
    },
    "Swimming": {
        "family": "Athletics",
        "prompts": [
            "Swimming",
            "swimming activity",
            "pool swimming",
            "swim practice"
        ],
        "modules": ["activity_bias"],
    },

    "Study": {
        "family": "Smarts",
        "prompts": [
            "Study",
            "studying",
            "studying at a desk",
            "schoolwork"
        ],
        "detector_weights": {"laptop": 3.5, "book": 4.0, "backpack": 1.0},
        "modules": ["activity_bias", "study_general"],
    },
    "School": {
        "family": "Smarts",
        "prompts": [
            "School",
            "school activity",
            "classroom learning",
            "student life"
        ],
        "detector_weights": {"book": 2.5, "backpack": 2.0},
        "modules": ["activity_bias", "school_like"],
    },
    "Coding": {
        "family": "Smarts",
        "prompts": [
            "Coding",
            "computer programming",
            "writing code",
            "software development"
        ],
        "detector_weights": {"laptop": 5.5},
        "modules": ["activity_bias", "screen_work"],
    },
    "Science": {
        "family": "Smarts",
        "prompts": [
            "Science",
            "science learning",
            "science activity",
            "scientific study"
        ],
        "modules": ["activity_bias"],
    },
    "Research": {
        "family": "Smarts",
        "prompts": [
            "Research",
            "academic research",
            "reading for research",
            "investigating information"
        ],
        "detector_weights": {"laptop": 2.5, "book": 2.5},
        "modules": ["activity_bias", "research_like"],
    },
    "Homework": {
        "family": "Smarts",
        "prompts": [
            "Homework",
            "doing homework",
            "student homework",
            "working on assignments"
        ],
        "modules": ["activity_bias", "homework_like"],
    },
    "Reading": {
        "family": "Smarts",
        "prompts": [
            "Reading",
            "reading a book",
            "studying from a book",
            "focused reading"
        ],
        "detector_weights": {"book": 4.0},
        "modules": ["activity_bias", "reading_like"],
    },
    "Math": {
        "family": "Smarts",
        "prompts": [
            "Math",
            "solving math problems",
            "math study session",
            "learning mathematics"
        ],
        "modules": ["activity_bias"],
    },
    "Exam": {
        "family": "Smarts",
        "prompts": [
            "Exam",
            "exam preparation",
            "testing session",
            "studying for an exam"
        ],
        "modules": ["activity_bias"],
    },
    "Project": {
        "family": "Smarts",
        "prompts": [
            "Project",
            "working on a project",
            "academic project",
            "building a project"
        ],
        "detector_weights": {"laptop": 2.5},
        "modules": ["activity_bias", "project_like"],
    },

    "Friends": {
        "family": "Social",
        "prompts": [
            "Friends",
            "friends together",
            "group of friends",
            "friend hangout"
        ],
        "detector_weights": {
            "dining table": 3.5,
            "cake": 3.0,
            "cup": 1.2,
            "bench": 0.8
        },
        "modules": ["activity_bias", "group_social"],
    },
    "Hangout": {
        "family": "Social",
        "prompts": [
            "Hangout",
            "casual hangout",
            "spending time together",
            "social hangout"
        ],
        "detector_weights": {
            "dining table": 4.0,
            "cup": 1.5,
            "chair": 0.8,
            "bench": 1.0
        },
        "modules": ["activity_bias", "hangout_like"],
    },
    "Family": {
        "family": "Social",
        "prompts": [
            "Family",
            "family moment",
            "family together",
            "time with family"
        ],
        "detector_weights": {
            "dining table": 3.5,
            "cake": 3.0
        },
        "modules": ["activity_bias", "family_like"],
    },
    "Party": {
        "family": "Social",
        "prompts": [
            "Party",
            "party scene",
            "celebration with friends",
            "social party"
        ],
        "detector_weights": {
            "cake": 6.0,
            "dining table": 2.5
        },
        "modules": ["activity_bias", "party_like"],
    },
    "Support": {
        "family": "Social",
        "prompts": [
            "Support",
            "helping someone",
            "being supportive",
            "supportive social moment"
        ],
        "modules": ["activity_bias", "supportive_social"],
    },
    "Volunteer": {
        "family": "Social",
        "prompts": [
            "Volunteer",
            "volunteer activity",
            "community helping",
            "group volunteering"
        ],
        "modules": ["activity_bias", "supportive_social"],
    },
    "Helping": {
        "family": "Social",
        "prompts": [
            "Helping",
            "helping activity",
            "helping others",
            "assisting someone"
        ],
        "modules": ["activity_bias", "supportive_social"],
    },

    "Beach": {
        "family": "Vibes",
        "prompts": [
            "Beach",
            "beach leisure",
            "relaxing at the beach",
            "beach day"
        ],
        "detector_weights": {
            "kite": 4.0,
            "boat": 2.0,
            "surfboard": 1.0
        },
        "modules": ["activity_bias", "beach_like"],
    },
    "Camping": {
        "family": "Vibes",
        "prompts": [
            "Camping",
            "camping trip",
            "outdoor camping",
            "camp leisure"
        ],
        "detector_weights": {"frisbee": 2.0},
        "modules": ["activity_bias"],
    },
    "Gaming": {
        "family": "Vibes",
        "prompts": [
            "Gaming",
            "playing video games",
            "gaming setup",
            "video game leisure"
        ],
        "modules": ["activity_bias", "screen_leisure"],
    },
    "Nature": {
        "family": "Vibes",
        "prompts": [
            "Nature",
            "nature scenery",
            "outdoor scenery",
            "time in nature"
        ],
        "detector_weights": {"dog": 2.0},
        "modules": ["activity_bias"],
    },
    "Pets": {
        "family": "Vibes",
        "prompts": [
            "Pets",
            "pet animal",
            "dog or cat",
            "pet moment"
        ],
        "detector_weights": {"dog": 8.0, "cat": 8.0},
        "modules": ["activity_bias", "pet_leisure"],
    },
    "Travel": {
        "family": "Vibes",
        "prompts": [
            "Travel",
            "travel moment",
            "trip activity",
            "vacation scene"
        ],
        "detector_weights": {
            "car": 2.5,
            "bus": 2.5,
            "train": 3.0,
            "airplane": 6.0,
            "boat": 4.5,
            "kite": 2.0
        },
        "modules": ["activity_bias", "travel_like"],
    },
    "Meditation": {
        "family": "Vibes",
        "prompts": [
            "Meditation",
            "meditating",
            "mindful relaxation",
            "quiet wellness moment"
        ],
        "modules": ["activity_bias"],
    },
    "Selfie": {
        "family": "Vibes",
        "prompts": [
            "Selfie",
            "taking a selfie",
            "casual selfie",
            "personal lifestyle photo"
        ],
        "detector_weights": {"cat": 1.5, "cell phone": 1.5},
        "modules": ["activity_bias", "selfie_like"],
    },
    "Walking": {
        "family": "Vibes",
        "prompts": [
            "Walking",
            "casual walking",
            "walk outdoors",
            "leisure walk"
        ],
        "detector_weights": {"dog": 2.0},
        "modules": ["activity_bias", "walking_like"],
    },

    "Art": {
        "family": "Creativity",
        "prompts": [
            "Art",
            "making art",
            "artistic creation",
            "art project"
        ],
        "modules": ["activity_bias"],
    },
    "Photography": {
        "family": "Creativity",
        "prompts": [
            "Photography",
            "taking photos",
            "photo activity",
            "camera photography"
        ],
        "detector_weights": {"cell phone": 1.5},
        "modules": ["activity_bias", "photography_like"],
    },
    "Music": {
        "family": "Creativity",
        "prompts": [
            "Music",
            "playing music",
            "musical activity",
            "performing music"
        ],
        "modules": ["activity_bias"],
    },
    "Painting": {
        "family": "Creativity",
        "prompts": [
            "Painting",
            "painting art",
            "making a painting",
            "paintbrush artwork"
        ],
        "modules": ["activity_bias"],
    },
    "Writing": {
        "family": "Creativity",
        "prompts": [
            "Writing",
            "creative writing",
            "writing activity",
            "writing on paper"
        ],
        "modules": ["activity_bias"],
    },
    "Drawing": {
        "family": "Creativity",
        "prompts": [
            "Drawing",
            "drawing art",
            "making a drawing",
            "sketching"
        ],
        "modules": ["activity_bias"],
    },
    "Dance": {
        "family": "Creativity",
        "prompts": [
            "Dance",
            "dancing performance",
            "dance activity",
            "creative movement"
        ],
        "modules": ["activity_bias"],
    },
    "Fashion": {
        "family": "Creativity",
        "prompts": [
            "Fashion",
            "fashion styling",
            "style expression",
            "fashion look"
        ],
        "detector_weights": {"cell phone": 0.8},
        "modules": ["activity_bias", "fashion_like"],
    },
    "Cooking": {
        "family": "Creativity",
        "prompts": [
            "Cooking",
            "cooking creatively",
            "food preparation",
            "making a meal"
        ],
        "modules": ["activity_bias"],
    },
}

FAMILY_TO_STAT = {family: family for family in FAMILY_SPECS.keys()}
TAG_TO_FAMILY = {tag: spec["family"] for tag, spec in TAG_SPECS.items()}
ACTIVITY_TAGS = set(TAG_SPECS.keys())