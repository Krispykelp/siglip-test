from .family_specs import FAMILY_SPECS

TAG_SPECS = {
    # =========================
    # ATHLETICS
    # =========================
    "Basketball": {
        "family": "Athletics",
        "prompts": [
            "Basketball",
            "basketball court with hoop and ball",
            "person dribbling a basketball",
            "basketball game on a court",
        ],
        "detector_weights": {"sports ball": 8.0},
        "modules": ["activity_bias", "ball_sport_like"],
    },
    "Cardio": {
        "family": "Athletics",
        "prompts": [
            "Cardio",
            "cardio workout indoors",
            "fitness cardio training session",
            "person doing cardio exercise",
        ],
        "modules": ["activity_bias", "cardio_like"],
    },
    "Workout": {
        "family": "Athletics",
        "prompts": [
            "Workout",
            "gym workout with weights or equipment",
            "person exercising in a gym",
            "bench press or fitness workout",
        ],
        "modules": ["activity_bias", "workout_like"],
    },
    "Hiking": {
        "family": "Athletics",
        "prompts": [
            "Hiking",
            "person hiking on a trail",
            "outdoor trail hiking",
            "hiker walking in nature",
        ],
        "modules": ["activity_bias"],
    },
    "Running": {
        "family": "Athletics",
        "prompts": [
            "Running",
            "person running outdoors",
            "marathon or running race",
            "runner on a road or track",
        ],
        "modules": ["activity_bias", "running_like"],
    },
    "Rock Climbing": {
        "family": "Athletics",
        "prompts": [
            "Rock Climbing",
            "person climbing a rock wall",
            "indoor climbing gym or climbing wall",
            "climber on a rock face or climbing holds",
        ],
        "modules": ["activity_bias", "rock_climbing_like"],
    },
    "Soccer": {
        "family": "Athletics",
        "prompts": [
            "Soccer",
            "soccer game on a field",
            "person playing soccer",
            "outdoor soccer match",
        ],
        "detector_weights": {"sports ball": 5.0},
        "modules": ["activity_bias", "ball_sport_like"],
    },
    "Cycling": {
        "family": "Athletics",
        "prompts": [
            "Cycling",
            "person riding a bicycle",
            "biking outdoors",
            "cycling on a road or trail",
        ],
        "detector_weights": {"bicycle": 8.0},
        "modules": ["activity_bias", "cycling_like"],
    },
    "Surfing": {
        "family": "Athletics",
        "prompts": [
            "Surfing",
            "person riding a surfboard",
            "surfer on a wave",
            "surfing on the water",
        ],
        "detector_weights": {"surfboard": 8.0},
        "modules": ["activity_bias", "surf_like"],
    },
    "Skateboarding": {
        "family": "Athletics",
        "prompts": [
            "Skateboarding",
            "person riding a skateboard",
            "skateboard trick outdoors",
            "skateboarding on pavement",
        ],
        "detector_weights": {"skateboard": 8.0},
        "modules": ["activity_bias", "skate_like"],
    },
    "Tennis": {
        "family": "Athletics",
        "prompts": [
            "Tennis",
            "person swinging a tennis racket",
            "tennis court with racket sport",
            "tennis match with a racket",
        ],
        "detector_weights": {"tennis racket": 8.0},
        "modules": ["activity_bias", "racket_sport_like"],
    },
    "Baseball": {
        "family": "Athletics",
        "prompts": [
            "Baseball",
            "baseball game on a field",
            "person playing baseball",
            "baseball bat and glove sport",
        ],
        "detector_weights": {
            "baseball bat": 8.0,
            "baseball glove": 7.0,
        },
        "modules": ["activity_bias", "baseball_like"],
    },
    "Skiing": {
        "family": "Athletics",
        "prompts": [
            "Skiing",
            "person skiing on snow",
            "skiing down a snowy slope",
            "winter ski activity",
        ],
        "detector_weights": {
            "skis": 8.0,
            "snowboard": 4.5,
        },
        "modules": ["activity_bias", "snow_sport_like"],
    },
    "Football": {
        "family": "Athletics",
        "prompts": [
            "Football",
            "football game on a field",
            "person playing football",
            "team football play",
        ],
        "detector_weights": {"sports ball": 5.0},
        "modules": ["activity_bias", "ball_sport_like"],
    },
    "Volleyball": {
        "family": "Athletics",
        "prompts": [
            "Volleyball",
            "people playing volleyball",
            "volleyball game",
            "beach or court volleyball",
        ],
        "detector_weights": {"sports ball": 5.0},
        "modules": ["activity_bias", "ball_sport_like"],
    },
    "Swimming": {
        "family": "Athletics",
        "prompts": [
            "Swimming",
            "person swimming in water",
            "pool swimming activity",
            "swimmer in a pool",
        ],
        "modules": ["activity_bias"],
    },

    # =========================
    # SMARTS
    # =========================
    "Study": {
        "family": "Smarts",
        "prompts": [
            "Study",
            "student studying with books and notes",
            "laptop and textbooks study session",
            "studying at a desk with school materials",
        ],
        "detector_weights": {"laptop": 3.5, "book": 4.0, "backpack": 1.0},
        "modules": ["activity_bias", "study_general"],
    },
    "School": {
        "family": "Smarts",
        "prompts": [
            "School",
            "student at school",
            "classroom school activity",
            "school learning environment",
        ],
        "detector_weights": {"book": 2.5, "backpack": 2.0},
        "modules": ["activity_bias", "school_like"],
    },
    "Coding": {
        "family": "Smarts",
        "prompts": [
            "Coding",
            "person writing code on a laptop",
            "computer programming work",
            "software development on a computer",
        ],
        "detector_weights": {"laptop": 5.5},
        "modules": ["activity_bias", "screen_work"],
    },
    "Science": {
        "family": "Smarts",
        "prompts": [
            "Science",
            "science learning activity",
            "person doing science work",
            "scientific study scene",
        ],
        "modules": ["activity_bias"],
    },
    "Research": {
        "family": "Smarts",
        "prompts": [
            "Research",
            "research papers and laptop on a desk",
            "academic research notes and articles",
            "person reading sources and taking research notes",
        ],
        "detector_weights": {"laptop": 2.5, "book": 2.5},
        "modules": ["activity_bias", "research_like"],
    },
    "Homework": {
        "family": "Smarts",
        "prompts": [
            "Homework",
            "student doing homework",
            "working on school assignments",
            "homework at a desk",
        ],
        "modules": ["activity_bias", "homework_like"],
    },
    "Reading": {
        "family": "Smarts",
        "prompts": [
            "Reading",
            "person reading a physical book",
            "open book held for reading",
            "quiet book reading scene",
        ],
        "detector_weights": {"book": 4.0},
        "modules": ["activity_bias", "reading_like"],
    },
    "Math": {
        "family": "Smarts",
        "prompts": [
            "Math",
            "solving math problems",
            "doing mathematics at a desk",
            "person working on math homework",
        ],
        "modules": ["activity_bias"],
    },
    "Exam": {
        "family": "Smarts",
        "prompts": [
            "Exam",
            "studying for an exam",
            "test preparation session",
            "person preparing for a test",
        ],
        "modules": ["activity_bias"],
    },
    "Project": {
        "family": "Smarts",
        "prompts": [
            "Project",
            "working on a project",
            "academic or technical project work",
            "building or preparing a project",
        ],
        "detector_weights": {"laptop": 2.5},
        "modules": ["activity_bias", "project_like"],
    },
    "Chess": {
        "family": "Smarts",
        "prompts": [
            "Chess",
            "person playing chess",
            "chess board with pieces",
            "strategy board game chess",
        ],
        "modules": ["activity_bias", "chess_like"],
    },

    # =========================
    # SOCIABILITY
    # =========================
    "Friends": {
        "family": "Sociability",
        "prompts": [
            "Friends",
            "group of friends together",
            "friends hanging out together",
            "casual friend group photo",
        ],
        "detector_weights": {
            "dining table": 3.5,
            "cake": 3.0,
            "cup": 1.2,
            "bench": 0.8,
        },
        "modules": ["activity_bias", "group_social"],
    },
    "Hangout": {
        "family": "Sociability",
        "prompts": [
            "Hangout",
            "casual hangout with other people",
            "relaxed time together",
            "people hanging out socially",
        ],
        "detector_weights": {
            "dining table": 4.0,
            "cup": 1.5,
            "chair": 0.8,
            "bench": 1.0,
        },
        "modules": ["activity_bias", "hangout_like"],
    },
    "Family": {
        "family": "Sociability",
        "prompts": [
            "Family",
            "family spending time together",
            "family group photo or moment",
            "parents kids or relatives together",
        ],
        "detector_weights": {
            "dining table": 3.5,
            "cake": 3.0,
        },
        "modules": ["activity_bias", "family_like"],
    },
    "Party": {
        "family": "Sociability",
        "prompts": [
            "Party",
            "birthday party with cake",
            "celebration with decorations and people",
            "group party scene",
        ],
        "detector_weights": {
            "cake": 6.0,
            "dining table": 2.5,
        },
        "modules": ["activity_bias", "party_like"],
    },
    "Support": {
        "family": "Sociability",
        "prompts": [
            "Support",
            "person comforting or supporting someone",
            "supportive interaction between people",
            "helpful social support moment",
        ],
        "modules": ["activity_bias", "supportive_social"],
    },
    "Volunteer": {
        "family": "Sociability",
        "prompts": [
            "Volunteer",
            "community volunteer activity",
            "people helping in an organized way",
            "group volunteering scene",
        ],
        "modules": ["activity_bias", "supportive_social"],
    },
    "Helping": {
        "family": "Sociability",
        "prompts": [
            "Helping",
            "person helping another person",
            "assisting someone with a task",
            "helpful interaction between people",
        ],
        "modules": ["activity_bias", "supportive_social"],
    },

    # =========================
    # VIBES
    # =========================
    "Beach": {
        "family": "Vibes",
        "prompts": [
            "Beach",
            "relaxing at the beach",
            "beach day by the ocean",
            "sand and ocean leisure scene",
        ],
        "detector_weights": {
            "kite": 4.0,
            "boat": 2.0,
            "surfboard": 1.0,
        },
        "modules": ["activity_bias", "beach_like"],
    },
    "Camping": {
        "family": "Vibes",
        "prompts": [
            "Camping",
            "camping outdoors",
            "campsite leisure scene",
            "camping trip in nature",
        ],
        "detector_weights": {"frisbee": 2.0},
        "modules": ["activity_bias"],
    },
    "Gaming": {
        "family": "Vibes",
        "prompts": [
            "Gaming",
            "person playing video games",
            "gaming setup indoors",
            "video game leisure scene",
        ],
        "modules": ["activity_bias", "screen_leisure"],
    },
    "Nature": {
        "family": "Vibes",
        "prompts": [
            "Nature",
            "mountain landscape scenery",
            "outdoor scenic view or wildlife",
            "nature photo with mountains or wilderness",
        ],
        "detector_weights": {"dog": 1.0},
        "modules": ["activity_bias", "nature_like"],
    },
    "Pets": {
        "family": "Vibes",
        "prompts": [
            "Pets",
            "person with a pet animal",
            "dog or cat companion moment",
            "pet centered lifestyle photo",
        ],
        "detector_weights": {"dog": 8.0, "cat": 8.0},
        "modules": ["activity_bias", "pet_leisure"],
    },
    "Travel": {
        "family": "Vibes",
        "prompts": [
            "Travel",
            "tourist standing by city landmark",
            "travel photo at famous tower or landmark",
            "city sightseeing scene",
        ],
        "detector_weights": {
            "car": 2.5,
            "bus": 2.5,
            "train": 3.0,
            "airplane": 6.0,
            "boat": 4.5,
            "kite": 2.0,
        },
        "modules": ["activity_bias", "travel_like"],
    },
    "Meditation": {
        "family": "Vibes",
        "prompts": [
            "Meditation",
            "person meditating calmly",
            "mindful seated relaxation",
            "quiet wellness meditation scene",
        ],
        "modules": ["activity_bias"],
    },
    "Selfie": {
        "family": "Vibes",
        "prompts": [
            "Selfie",
            "person taking a selfie",
            "phone facing self portrait",
            "casual selfie lifestyle photo",
        ],
        "detector_weights": {"cat": 1.5, "cell phone": 1.5},
        "modules": ["activity_bias", "selfie_like"],
    },
    "Walking": {
        "family": "Vibes",
        "prompts": [
            "Walking",
            "person walking casually outdoors",
            "leisure walk outside",
            "casual outdoor walking scene",
        ],
        "detector_weights": {"dog": 2.0},
        "modules": ["activity_bias", "walking_like"],
    },

    # =========================
    # CREATIVITY
    # =========================
    "Art": {
        "family": "Creativity",
        "prompts": [
            "Art",
            "person making visual art",
            "handmade art creation",
            "art project in progress",
        ],
        "modules": ["activity_bias"],
    },
    "Photography": {
        "family": "Creativity",
        "prompts": [
            "person holding a camera up to take a photo",
            "phone raised in front of face to photograph",
            "photographer aiming a camera or phone at a subject",
            "visible camera or phone used to take a picture",
        ],
        "detector_weights": {"cell phone": 0.8},
        "modules": ["activity_bias", "photography_like"],
    },
    "Music": {
        "family": "Creativity",
        "prompts": [
            "music studio with microphone and instrument",
            "person playing guitar or piano in studio",
            "recording session with musician and microphone",
            "musician performing with instrument",
        ],
        "modules": ["activity_bias", "music_like"],
    },
    "Painting": {
        "family": "Creativity",
        "prompts": [
            "paintbrush painting on canvas",
            "artist painting at an easel",
            "wet paint on canvas artwork",
            "canvas painting with visible brush strokes",
        ],
        "modules": ["activity_bias", "painting_like"],
    },
    "Writing": {
        "family": "Creativity",
        "prompts": [
            "handwriting in a notebook",
            "person writing on paper at a desk",
            "writing notes in a notebook",
            "hand writing on paper",
        ],
        "modules": ["activity_bias", "writing_like"],
    },
    "Drawing": {
        "family": "Creativity",
        "prompts": [
            "pencil sketch on paper",
            "hand drawing in a sketchbook",
            "drawing with pencil on paper",
            "line art sketch on notebook page",
        ],
        "modules": ["activity_bias", "drawing_like"],
    },
    "Dance": {
        "family": "Creativity",
        "prompts": [
            "Dance",
            "person dancing in motion",
            "dance performance scene",
            "creative dance movement",
        ],
        "modules": ["activity_bias"],
    },
    "Fashion": {
        "family": "Creativity",
        "prompts": [
            "person showing an outfit style",
            "fashion pose or mirror style photo",
            "style expression through clothing",
            "outfit mirror photo or fashion pose",
        ],
        "detector_weights": {"cell phone": 0.8},
        "modules": ["activity_bias", "fashion_like"],
    },
    "Cooking": {
        "family": "Creativity",
        "prompts": [
            "person preparing food in a kitchen",
            "cooking a meal",
            "food preparation scene",
            "kitchen cooking activity",
        ],
        "modules": ["activity_bias"],
    },
}

FAMILY_TO_STAT = {family: family for family in FAMILY_SPECS.keys()}
TAG_TO_FAMILY = {tag: spec["family"] for tag, spec in TAG_SPECS.items()}
ACTIVITY_TAGS = set(TAG_SPECS.keys())