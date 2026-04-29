import os
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from ultralytics import YOLO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# --------------------------------------------------
# Config
# --------------------------------------------------
VLM_MODEL_NAME = "google/siglip2-base-patch16-224"
YOLO_MODEL_NAME = "yolo11n.pt"
IMAGE_PATH = "test.jpg"

CLAIMED_TAGS = ["Study", "Coding"]

TOP_FAMILIES_TO_KEEP = 1
TOP_K_CANONICAL = 12

SUPPORT_MARGIN_FROM_TOP = 1.2
MIN_SUPPORT_SCORE = -12.0

FAMILY_SUPPORT_MARGIN_FROM_TOP = 25.0
FAMILY_SUPPORT_MAX_DROP_FROM_DIRECT = 25.0

MAX_BONUS_TAGS = 2
BONUS_MARGIN_FROM_TOP = 2.0
MIN_BONUS_SCORE = -20.0

ACTIVITY_BIAS = 0.40

FAMILY_MIN_SCORE = -14.0
FAMILY_MIN_MARGIN = 1.5
CREATIVITY_FAMILY_MIN_SCORE = -18.5
CREATIVITY_FAMILY_MIN_MARGIN = 0.25
DETECTOR_OVERRIDE_FAMILIES = {"Athletics", "Smarts", "Social", "Vibes"}

DIRECT_REWARD = 2
FAMILY_SUPPORT_REWARD = 1
PRIMARY_INFERRED_REWARD = 2
BONUS_REWARD = 1

# --------------------------------------------------
# Trusted detector classes
# --------------------------------------------------
TRUSTED_DETECTOR_CLASSES = {
    "person",
    "sports ball",
    "bicycle",
    "surfboard",
    "skateboard",
    "tennis racket",
    "dog",
    "cat",
    "frisbee",
    "baseball bat",
    "baseball glove",
    "skis",
    "snowboard",
    "kite",
    "laptop",
    "cell phone",
    "book",
    "backpack",
    "chair",
    "bench",
    "car",
    "bus",
    "train",
    "airplane",
    "boat",
    "cup",
    "dining table",
    "cake",
}

# --------------------------------------------------
# Family prompt specs
# --------------------------------------------------
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
            "friends spending time together",
            "social gathering with clear interaction",
            "celebration with friends or family",
            "people interacting in a social setting",
            "family or friends sharing a moment"
        ],
        "detector_weights": {
            "dining table": 2.0,
            "cake": 2.8,
            "cup": 0.5,
            "chair": 0.2,
            "bench": 0.2,
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
            "drawing painting or sketching",
            "music performance or instrument practice",
            "photography or taking a photo",
            "creative project or studio work",
            "visual creativity and self expression"
        ],
        "detector_weights": {
            "cell phone": 0.3,
        },
        "modules": ["creativity_family"]
    }
}

# --------------------------------------------------
# Tag specs
# Most future scaling should happen here.
# --------------------------------------------------
TAG_SPECS = {
    # ---------------- Athletics ----------------
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
    "Cardio": {
        "family": "Athletics",
        "prompts": [
            "Cardio",
            "cardio workout",
            "treadmill cardio",
            "indoor exercise session",
            "person doing cardio fitness"
        ],
        "modules": ["activity_bias", "workout_like"],
    },
    "Hiking": {
        "family": "Athletics",
        "prompts": [
            "Hiking",
            "outdoor hiking",
            "trail walking",
            "hiking outdoors",
            "person hiking on a trail"
        ],
        "modules": ["activity_bias", "hiking_like"],
    },
    "Rock Climbing": {
        "family": "Athletics",
        "prompts": [
            "Rock Climbing",
            "person climbing a rock wall",
            "indoor climbing gym",
            "outdoor rock climbing",
            "climber on a wall"
        ],
        "modules": ["activity_bias", "climbing_like"],
    },
    "Running": {
        "family": "Athletics",
        "prompts": [
            "Running",
            "outdoor running",
            "jogging",
            "running exercise"
        ],
        "modules": ["activity_bias", "running_like"],
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

    # ---------------- Smarts ----------------
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

    # ---------------- Social ----------------
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

    # ---------------- Vibes ----------------
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
            "mountain or forest scenery",
            "wildlife or outdoor nature",
            "time in nature"
        ],
        "detector_weights": {"dog": 2.0, "cat": 1.5},
        "modules": ["activity_bias", "nature_like"],
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
            "vacation scene",
            "famous landmark while traveling",
            "tourist city scene"
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

    # ---------------- Creativity ----------------
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
            "camera photography",
            "person taking a picture",
            "photographer framing a shot"
        ],
        "detector_weights": {"cell phone": 1.2},
        "modules": ["activity_bias", "photography_like"],
    },
    "Music": {
        "family": "Creativity",
        "prompts": [
            "Music",
            "playing music",
            "musical activity",
            "performing music",
            "person holding a guitar or instrument",
            "orchestra or music studio"
        ],
        "modules": ["activity_bias", "music_like"],
    },
    "Painting": {
        "family": "Creativity",
        "prompts": [
            "Painting",
            "painting art",
            "making a painting",
            "paintbrush artwork",
            "artist painting on canvas",
            "easel and paint palette"
        ],
        "modules": ["activity_bias", "painting_like"],
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
            "sketching",
            "person sketching on paper",
            "pencil drawing in sketchbook"
        ],
        "modules": ["activity_bias", "drawing_like"],
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

# --------------------------------------------------
# Image / model helpers
# --------------------------------------------------
def print_device_info():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using device: cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version (torch build): {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
        print("WARNING: CUDA not available. Running on CPU.")
    return device


def load_pil_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find image at: {os.path.abspath(image_path)}")
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"PIL could not open image: {os.path.abspath(image_path)}\n{e}")


def validate_image_for_pipeline(image_path: str) -> None:
    pil_img = load_pil_image(image_path)
    try:
        arr = np.array(pil_img)
        if arr is None or arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError(f"Expected RGB image [H, W, 3], got shape {arr.shape}")
    finally:
        pil_img.close()


def load_vlm(model_name, device):
    print("Loading VLM...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def load_yolo(model_name):
    print("Loading YOLO detector...")
    return YOLO(model_name)

# --------------------------------------------------
# Prompt helpers
# --------------------------------------------------
def flatten_simple_prompt_map(prompt_map):
    all_prompts = []
    prompt_to_key = []
    for key, prompts in prompt_map.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_key.append(key)
    return all_prompts, prompt_to_key


def flatten_nested_prompt_map(nested_map):
    all_prompts = []
    prompt_to_canonical = []
    for canonical_tag, prompts in nested_map.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_canonical.append(canonical_tag)
    return all_prompts, prompt_to_canonical

# --------------------------------------------------
# Inference helpers
# --------------------------------------------------
def run_vlm(processor, model, device, image_path, all_prompts):
    image = load_pil_image(image_path)
    inputs = processor(
        text=all_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.time()

    scores = outputs.logits_per_image.detach().cpu().numpy()[0].tolist()
    image.close()
    return scores, end - start


def collapse_scores(keys, scores, prompts):
    collapsed = {}
    for key, score, prompt in zip(keys, scores, prompts):
        if key not in collapsed or score > collapsed[key]["score"]:
            collapsed[key] = {"score": score, "best_prompt": prompt}
    return collapsed


def sort_collapsed(collapsed):
    results = []
    for key, info in collapsed.items():
        results.append((key, info["score"], info["best_prompt"]))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def run_yolo_detection(detector, image_path):
    pil_img = load_pil_image(image_path)
    rgb_array = np.array(pil_img)
    pil_img.close()

    if rgb_array is None or rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise RuntimeError(f"YOLO input invalid: {os.path.abspath(image_path)}")

    start = time.time()
    results = detector(rgb_array, verbose=False)
    end = time.time()

    names = results[0].names
    boxes = results[0].boxes

    detections = []
    if boxes is not None and boxes.cls is not None:
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        confs = boxes.conf.detach().cpu().numpy().tolist()
        for cls_id, conf in zip(cls_ids, confs):
            detections.append({
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": conf
            })

    return detections, end - start


def summarize_detections(detections):
    return Counter(d["class_name"] for d in detections)


def summarize_trusted_detections(detections):
    trusted = [d for d in detections if d["class_name"] in TRUSTED_DETECTOR_CLASSES]
    return Counter(d["class_name"] for d in trusted), trusted

# --------------------------------------------------
# Context
# --------------------------------------------------
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

# --------------------------------------------------
# Shared scoring primitives
# --------------------------------------------------
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

# --------------------------------------------------
# Reusable module system
# --------------------------------------------------
def apply_module(module_name, adjusted, adjustments, detection_counts, ctx, entity_name, entity_type):
    if module_name == "activity_bias":
        adjusted += ACTIVITY_BIAS
        add_adjustment(adjustments, "global:activity_bias", ACTIVITY_BIAS)

    # ---------- family modules ----------
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
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:pair_scene", 0.8)
        if ctx["group_scene"] and not ctx["sports_scene"] and not ctx["study_scene"]:
            adjusted += 1.0
            add_adjustment(adjustments, "ctx:group_scene", 1.0)
        if ctx["structured_social_scene"]:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:structured_social_scene", 1.2)
        if detection_counts.get("cake", 0) > 0 and detection_counts.get("person", 0) >= 2:
            adjusted += 2.2
            add_adjustment(adjustments, "ctx:cake+group", 2.2)
        if ctx["sports_scene"] or ctx["study_scene"]:
            adjusted -= 3.0
            add_adjustment(adjustments, "penalty:non_social_activity_present", -3.0)
        if ctx["study_scene"] and ctx["group_scene"]:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:group_study_scene", -1.0)
        if ctx["sports_scene"] and ctx["group_scene"]:
            adjusted -= 1.5
            add_adjustment(adjustments, "penalty:team_sports_scene", -1.5)
        if ctx["solo_person"]:
            adjusted -= 2.2
            add_adjustment(adjustments, "penalty:solo_person", -2.2)
        if not ctx["structured_social_scene"] and detection_counts.get("cake", 0) == 0 and detection_counts.get("dining table", 0) == 0:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:no_clear_social_props", -1.2)
        if not ctx["large_group_scene"] and detection_counts.get("cup", 0) == 0 and detection_counts.get("bench", 0) == 0:
            adjusted -= 0.8
            add_adjustment(adjustments, "penalty:no_secondary_social_props", -0.8)

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
        if not ctx["sports_scene"] and not ctx["study_scene"] and not ctx["group_scene"]:
            adjusted += 1.5
            add_adjustment(adjustments, "ctx:default_vibes_boost", 1.5)
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
            adjusted += 0.3
            add_adjustment(adjustments, "ctx:cell_phone_present", 0.3)
        if detection_counts.get("person", 0) == 1 and detection_counts.get("cell phone", 0) > 0:
            adjusted += 0.4
            add_adjustment(adjustments, "ctx:solo_phone_scene", 0.4)
        if not ctx["sports_scene"] and not ctx["study_scene"] and not ctx["structured_social_scene"]:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:open_creative_scene", 1.2)
        if not ctx["sports_scene"] and not ctx["study_scene"]:
            adjusted += 1.5
            add_adjustment(adjustments, "ctx:default_creativity_bias", 1.5)
        if detection_counts.get("person", 0) <= 1 and not ctx["travel_scene"] and not ctx["pet_scene"]:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:solo_or_object_focus_scene", 0.8)
        if ctx["sports_scene"]:
            adjusted -= 1.5
            add_adjustment(adjustments, "penalty:sports_scene", -1.5)
        if ctx["study_scene"]:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:study_scene", -1.0)

    # ---------- tag modules ----------
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
        if detection_counts.get("surfboard", 0) > 0 or detection_counts.get("kite", 0) > 0 or detection_counts.get("boat", 0) > 0:
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
            adjusted += 1.5
            add_adjustment(adjustments, "ctx:phone_for_photography", 1.5)
        if detection_counts.get("person", 0) <= 1 and not ctx["study_scene"] and not ctx["sports_scene"]:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:single_subject_for_photography", 0.8)
        if ctx["travel_scene"] or ctx["pet_scene"]:
            adjusted += 0.6
            add_adjustment(adjustments, "ctx:event_or_subject_capture_for_photography", 0.6)

    elif module_name == "drawing_like":
        if not ctx["sports_scene"] and not ctx["study_scene"] and not ctx["structured_social_scene"]:
            adjusted += 1.5
            add_adjustment(adjustments, "ctx:creative_focus_for_drawing", 1.5)
        if detection_counts.get("person", 0) <= 1:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:low_person_count_for_drawing", 0.8)
        if detection_counts.get("person", 0) >= 1:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:person_for_drawing", 1.2)

    elif module_name == "painting_like":
        if not ctx["sports_scene"] and not ctx["study_scene"] and not ctx["structured_social_scene"]:
            adjusted += 1.5
            add_adjustment(adjustments, "ctx:creative_focus_for_painting", 1.5)
        if detection_counts.get("person", 0) <= 1:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:low_person_count_for_painting", 0.8)
        if detection_counts.get("person", 0) >= 1:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:person_for_painting", 1.2)

    elif module_name == "music_like":
        if not ctx["sports_scene"] and not ctx["study_scene"]:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:non_sports_non_study_for_music", 1.2)
        if ctx["structured_social_scene"]:
            adjusted -= 0.6
            add_adjustment(adjustments, "penalty:structured_social_scene", -0.6)
        if detection_counts.get("person", 0) == 0:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:no_person_for_music", -1.0)
        if (not ctx["group_scene"] and not ctx["structured_social_scene"] and detection_counts.get("cell phone", 0) == 0):
            adjusted -= 0.8
            add_adjustment(adjustments, "penalty:no_music_context", -0.8)
        if ctx["travel_scene"] or ctx["sports_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:travel_or_sports_scene", -1.2)

    elif module_name == "nature_like":
        if detection_counts.get("person", 0) <= 1 and not ctx["study_scene"] and not ctx["structured_social_scene"]:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:scenery_or_wildlife_scene", 1.2)
        if ctx["travel_scene"]:
            adjusted += 0.5
            add_adjustment(adjustments, "ctx:travel_overlap_for_nature", 0.5)
        if not ctx["sports_scene"] and not ctx["study_scene"] and not ctx["structured_social_scene"]:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:quiet_outdoor_scene_for_nature", 0.8)

    elif module_name == "climbing_like":
        if detection_counts.get("person", 0) >= 1 and not ctx["study_scene"]:
            adjusted += 1.8
            add_adjustment(adjustments, "ctx:person_for_climbing", 1.8)
        if not ctx["study_scene"] and not ctx["structured_social_scene"]:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:activity_focus_for_climbing", 0.8)
        if ctx["structured_social_scene"]:
            adjusted -= 0.8
            add_adjustment(adjustments, "penalty:structured_social_scene", -0.8)

    elif module_name == "hiking_like":
        if detection_counts.get("person", 0) >= 1 and not ctx["study_scene"]:
            adjusted += 1.0
            add_adjustment(adjustments, "ctx:person_for_hiking", 1.0)
        if not ctx["study_scene"] and not ctx["social_scene_props"]:
            adjusted += 1.3
            add_adjustment(adjustments, "ctx:outdoor_activity", 1.3)
        if ctx["travel_scene"] and detection_counts.get("person", 0) <= 2:
            adjusted += 0.6
            add_adjustment(adjustments, "ctx:outdoor_travel_overlap", 0.6)

    elif module_name == "running_like":
        if detection_counts.get("person", 0) >= 1:
            adjusted += 1.2
            add_adjustment(adjustments, "ctx:person_for_running", 1.2)
        if ctx["group_scene"]:
            adjusted += 1.5
            add_adjustment(adjustments, "ctx:group_running_scene", 1.5)
        if ctx["structured_social_scene"]:
            adjusted -= 1.0
            add_adjustment(adjustments, "penalty:structured_social_scene", -1.0)

    elif module_name == "fashion_like":
        if detection_counts.get("cell phone", 0) > 0 and detection_counts.get("person", 0) >= 1:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:person+phone_for_fashion", 0.8)

    elif module_name == "ball_sport_like":
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "workout_like":
        if detection_counts.get("person", 0) >= 1:
            adjusted += 1.5
            add_adjustment(adjustments, "ctx:person_present_for_workout", 1.5)
        if ctx["solo_person"] or ctx["group_scene"]:
            adjusted += 1.5
            add_adjustment(adjustments, "ctx:person_for_workout", 1.5)
        if detection_counts.get("person", 0) >= 2:
            adjusted += 0.8
            add_adjustment(adjustments, "ctx:workout_multi_person", 0.8)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)
        if ctx["structured_social_scene"]:
            adjusted -= 0.8
            add_adjustment(adjustments, "penalty:structured_social_scene", -0.8)

    elif module_name == "cycling_like":
        adjusted += 2.0
        add_adjustment(adjustments, "ctx:bicycle_for_cycling", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "surf_like":
        adjusted += 2.0
        add_adjustment(adjustments, "ctx:surfboard_for_surfing", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "skate_like":
        adjusted += 2.0
        add_adjustment(adjustments, "ctx:skateboard_for_skateboarding", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "racket_sport_like":
        adjusted += 2.0
        add_adjustment(adjustments, "ctx:racket_for_tennis", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "baseball_like":
        adjusted += 2.0
        add_adjustment(adjustments, "ctx:baseball_gear", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    elif module_name == "snow_sport_like":
        adjusted += 2.0
        add_adjustment(adjustments, "ctx:snow_sport_gear", 2.0)
        if ctx["study_scene"]:
            adjusted -= 1.2
            add_adjustment(adjustments, "penalty:study_scene", -1.2)

    else:
        add_adjustment(adjustments, f"warning:unknown_module:{module_name}", 0.0)

    return adjusted

# --------------------------------------------------
# Family evaluation
# --------------------------------------------------
def build_family_evidence(family, base_score, detection_counts, ctx):
    spec = FAMILY_SPECS[family]
    adjustments = []
    adjusted = base_score

    adjusted = apply_weight_map(
        adjusted, adjustments,
        spec.get("detector_weights", {}),
        detection_counts,
        "detector"
    )

    for module_name in spec.get("modules", []):
        adjusted = apply_module(
            module_name, adjusted, adjustments,
            detection_counts, ctx,
            entity_name=family, entity_type="family"
        )

    return {
        "family": family,
        "base_score": base_score,
        "final_score": adjusted,
        "best_prompt": None,
        "adjustments": adjustments
    }


def summarize_family_decision(family_results_with_evidence, forced_family):
    if not family_results_with_evidence:
        return {
            "winner": None,
            "runner_up": None,
            "margin": None,
            "forced_family": forced_family,
            "reason": "no_family_results"
        }

    winner = family_results_with_evidence[0]
    runner_up = family_results_with_evidence[1] if len(family_results_with_evidence) > 1 else None
    margin = None if runner_up is None else winner["final_score"] - runner_up["final_score"]

    if forced_family is not None:
        reason = f"detector override selected {forced_family}"
    elif runner_up is None:
        reason = "only one viable family"
    else:
        reason = f"{winner['family']} beat {runner_up['family']} by {margin:.3f}"

    return {
        "winner": winner["family"],
        "runner_up": runner_up["family"] if runner_up else None,
        "margin": margin,
        "forced_family": forced_family,
        "reason": reason
    }


def detection_override_family(detection_counts, ctx):
    if detection_counts.get("sports ball", 0) > 0 and detection_counts.get("person", 0) >= 1:
        return "Athletics"
    if detection_counts.get("bicycle", 0) > 0:
        return "Athletics"
    if detection_counts.get("surfboard", 0) > 0:
        return "Athletics"
    if detection_counts.get("skateboard", 0) > 0:
        return "Athletics"
    if detection_counts.get("tennis racket", 0) > 0:
        return "Athletics"
    if detection_counts.get("baseball bat", 0) > 0 or detection_counts.get("baseball glove", 0) > 0:
        return "Athletics"
    if detection_counts.get("skis", 0) > 0 or detection_counts.get("snowboard", 0) > 0:
        return "Athletics"

    if detection_counts.get("laptop", 0) >= 1 and detection_counts.get("person", 0) >= 1:
        return "Smarts"
    if detection_counts.get("book", 0) >= 2 and detection_counts.get("person", 0) >= 1:
        return "Smarts"

    if ctx["structured_social_scene"] and not ctx["sports_scene"] and not ctx["study_scene"]:
        return "Social"
    if ctx["large_group_scene"] and not ctx["sports_scene"] and not ctx["study_scene"]:
        return "Social"
    if detection_counts.get("cake", 0) > 0 and detection_counts.get("person", 0) >= 2:
        return "Social"

    if ctx["pet_scene"] and detection_counts.get("person", 0) <= 1:
        return "Vibes"
    if ctx["travel_scene"] and detection_counts.get("person", 0) <= 2 and not ctx["sports_scene"]:
        return "Vibes"

    return None


def apply_family_fusion(family_results, detection_counts):
    ctx = infer_context_signals(detection_counts)
    traced = []

    for family, score, best_prompt in family_results:
        evidence = build_family_evidence(
            family=family,
            base_score=score,
            detection_counts=detection_counts,
            ctx=ctx
        )
        evidence["best_prompt"] = best_prompt
        traced.append(evidence)

    traced.sort(key=lambda x: x["final_score"], reverse=True)

    forced_family = detection_override_family(detection_counts, ctx)
    if forced_family is not None:
        forced_row = None
        others = []
        for row in traced:
            if row["family"] == forced_family:
                forced_row = row
            else:
                others.append(row)
        if forced_row is not None:
            traced = [forced_row] + others

    compact = [(row["family"], row["final_score"], row["best_prompt"]) for row in traced]
    decision_summary = summarize_family_decision(traced, forced_family)
    return compact, traced, ctx, decision_summary


def family_confidence_passes(family_results, forced_family):
    if forced_family in DETECTOR_OVERRIDE_FAMILIES:
        return True, f"detector_override:{forced_family}"

    if len(family_results) == 0:
        return False, "no_family_results"

    top_family, top_score, _ = family_results[0]
    second_score = family_results[1][1] if len(family_results) > 1 else float("-inf")
    margin = float("inf") if second_score == float("-inf") else (top_score - second_score)

    min_score = FAMILY_MIN_SCORE
    min_margin = FAMILY_MIN_MARGIN

    if top_family == "Creativity":
        min_score = CREATIVITY_FAMILY_MIN_SCORE
        min_margin = CREATIVITY_FAMILY_MIN_MARGIN

    if top_score < min_score:
        return False, f"top_family_score_too_low:{top_family}:{top_score:.6f}"

    if len(family_results) > 1 and margin < min_margin:
        return False, f"top_family_margin_too_small:{top_family}:{margin:.6f}"

    return True, "score_margin_pass"

# --------------------------------------------------
# Tag evaluation
# --------------------------------------------------
def build_tag_evidence(tag, base_score, detection_counts, ctx):
    spec = TAG_SPECS[tag]
    adjustments = []
    adjusted = base_score

    adjusted = apply_weight_map(
        adjusted, adjustments,
        spec.get("detector_weights", {}),
        detection_counts,
        "detector"
    )

    adjusted = apply_context_weight_map(
        adjusted, adjustments,
        spec.get("context_weights", {}),
        ctx,
        "ctx"
    )

    for module_name in spec.get("modules", []):
        adjusted = apply_module(
            module_name, adjusted, adjustments,
            detection_counts, ctx,
            entity_name=tag, entity_type="tag"
        )

    return {
        "tag": tag,
        "family": spec["family"],
        "base_score": base_score,
        "final_score": adjusted,
        "best_prompt": None,
        "adjustments": adjustments
    }


def summarize_tag_decision(traced_tag_results):
    if not traced_tag_results:
        return {
            "winner": None,
            "runner_up": None,
            "margin": None,
            "reason": "no_tag_results"
        }

    winner = traced_tag_results[0]
    runner_up = traced_tag_results[1] if len(traced_tag_results) > 1 else None
    margin = None if runner_up is None else winner["final_score"] - runner_up["final_score"]

    if runner_up is None:
        reason = "only one viable tag"
    else:
        reason = f"{winner['tag']} beat {runner_up['tag']} by {margin:.3f}"

    return {
        "winner": winner["tag"],
        "runner_up": runner_up["tag"] if runner_up else None,
        "margin": margin,
        "reason": reason
    }


def apply_tag_fusion(canonical_results, detection_counts):
    ctx = infer_context_signals(detection_counts)
    traced = []

    for tag, score, best_prompt in canonical_results:
        evidence = build_tag_evidence(
            tag=tag,
            base_score=score,
            detection_counts=detection_counts,
            ctx=ctx
        )
        evidence["best_prompt"] = best_prompt
        traced.append(evidence)

    traced.sort(key=lambda x: x["final_score"], reverse=True)

    compact = [(row["tag"], row["final_score"], row["best_prompt"]) for row in traced]
    decision_summary = summarize_tag_decision(traced)
    return compact, traced, decision_summary

# --------------------------------------------------
# Support / inference logic
# --------------------------------------------------
def is_supported_claimed(score, top_score):
    return score >= MIN_SUPPORT_SCORE and (top_score - score) <= SUPPORT_MARGIN_FROM_TOP


def is_family_supported_claimed(claimed_tag, top_family, score, top_score, best_direct_score):
    tag_family = TAG_TO_FAMILY.get(claimed_tag)
    if tag_family != top_family:
        return False
    if score is None:
        return False
    if (top_score - score) > FAMILY_SUPPORT_MARGIN_FROM_TOP:
        return False
    if best_direct_score is not None and (best_direct_score - score) > FAMILY_SUPPORT_MAX_DROP_FROM_DIRECT:
        return False
    return True


def select_primary_and_bonus_inferred(canonical_results, claimed_tags, top_family):
    primary_inferred = None
    bonus_inferred = []

    if not canonical_results:
        return primary_inferred, bonus_inferred

    top_tag, top_score, top_prompt = canonical_results[0]

    if TAG_TO_FAMILY.get(top_tag) != top_family:
        return primary_inferred, bonus_inferred

    if len(claimed_tags) == 0:
        primary_inferred = (top_tag, top_score, top_prompt)

        for tag, score, best_prompt in canonical_results[1:]:
            if TAG_TO_FAMILY.get(tag) != top_family:
                continue
            if score < MIN_BONUS_SCORE:
                continue
            if (top_score - score) > BONUS_MARGIN_FROM_TOP:
                continue
            bonus_inferred.append((tag, score, best_prompt))
    else:
        for tag, score, best_prompt in canonical_results:
            if tag in claimed_tags:
                continue
            if TAG_TO_FAMILY.get(tag) != top_family:
                continue
            if score < MIN_BONUS_SCORE:
                continue
            if (top_score - score) > BONUS_MARGIN_FROM_TOP:
                continue
            bonus_inferred.append((tag, score, best_prompt))

    return primary_inferred, bonus_inferred[:MAX_BONUS_TAGS]


def build_verifier_summary(canonical_results, claimed_tags, top_family):
    top_score = canonical_results[0][1] if canonical_results else float("-inf")
    score_lookup = {tag: score for tag, score, _ in canonical_results}

    direct_supported = []
    family_supported = []
    unsupported = []

    for claimed in claimed_tags:
        score = score_lookup.get(claimed)
        if score is None:
            unsupported.append((claimed, None))
        elif is_supported_claimed(score, top_score):
            direct_supported.append((claimed, score))
        else:
            unsupported.append((claimed, score))

    best_direct_score = max((score for _, score in direct_supported), default=None)

    remaining = []
    for claimed, score in unsupported:
        if score is not None and is_family_supported_claimed(
            claimed_tag=claimed,
            top_family=top_family,
            score=score,
            top_score=top_score,
            best_direct_score=best_direct_score
        ):
            family_supported.append((claimed, score))
        else:
            remaining.append((claimed, score))

    unsupported = remaining
    primary_inferred, bonus_inferred = select_primary_and_bonus_inferred(
        canonical_results=canonical_results,
        claimed_tags=claimed_tags,
        top_family=top_family
    )

    return direct_supported, family_supported, unsupported, primary_inferred, bonus_inferred


def resolve_rewards(direct_supported, family_supported, primary_inferred, bonus_inferred):
    family_best = {}

    def try_assign(tag, score, reward_type, amount):
        family = TAG_TO_FAMILY.get(tag)
        if family is None:
            return

        candidate = {
            "family": family,
            "stat": FAMILY_TO_STAT[family],
            "tag": tag,
            "score": score,
            "reward_type": reward_type,
            "amount": amount
        }

        current = family_best.get(family)
        priority = {
            "direct": 4,
            "family": 3,
            "primary_inferred": 2,
            "bonus": 1
        }

        if current is None:
            family_best[family] = candidate
            return

        current_pri = priority[current["reward_type"]]
        candidate_pri = priority[candidate["reward_type"]]

        if candidate_pri > current_pri:
            family_best[family] = candidate
        elif candidate_pri == current_pri and candidate["score"] > current["score"]:
            family_best[family] = candidate

    for tag, score in direct_supported:
        try_assign(tag, score, "direct", DIRECT_REWARD)

    for tag, score in family_supported:
        try_assign(tag, score, "family", FAMILY_SUPPORT_REWARD)

    if primary_inferred is not None:
        tag, score, _ = primary_inferred
        try_assign(tag, score, "primary_inferred", PRIMARY_INFERRED_REWARD)

    for tag, score, _ in bonus_inferred:
        try_assign(tag, score, "bonus", BONUS_REWARD)

    return list(family_best.values())

# --------------------------------------------------
# Printing
# --------------------------------------------------
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

# --------------------------------------------------
# Benchmark / reusable pipeline API
# --------------------------------------------------
_PIPELINE_CACHE = None


def get_pipeline():
    """
    Load models once and reuse them across benchmark images.
    Returns a cached dict with device, processor, vlm, and detector.
    """
    global _PIPELINE_CACHE

    if _PIPELINE_CACHE is not None:
        return _PIPELINE_CACHE

    device = print_device_info()
    processor, vlm = load_vlm(VLM_MODEL_NAME, device)
    detector = load_yolo(YOLO_MODEL_NAME)

    _PIPELINE_CACHE = {
        "device": device,
        "processor": processor,
        "vlm": vlm,
        "detector": detector,
    }
    return _PIPELINE_CACHE


def analyze_image(image_path: str, top_k_tags: int = 3):
    """
    Runs the current SigLIP + YOLO pipeline on one image and returns
    structured output for benchmark.py.
    """
    validate_image_for_pipeline(image_path)

    pipeline = get_pipeline()
    device = pipeline["device"]
    processor = pipeline["processor"]
    vlm = pipeline["vlm"]
    detector = pipeline["detector"]

    detections, _ = run_yolo_detection(detector, image_path)
    all_detection_counts = summarize_detections(detections)
    trusted_detection_counts, _ = summarize_trusted_detections(detections)

    family_prompt_map = {family: spec["prompts"] for family, spec in FAMILY_SPECS.items()}
    family_prompts, family_keys = flatten_simple_prompt_map(family_prompt_map)
    family_scores, _ = run_vlm(
        processor=processor,
        model=vlm,
        device=device,
        image_path=image_path,
        all_prompts=family_prompts
    )

    collapsed_families = collapse_scores(
        keys=family_keys,
        scores=family_scores,
        prompts=family_prompts
    )
    family_results = sort_collapsed(collapsed_families)
    family_results, traced_family_results, ctx, family_decision_summary = apply_family_fusion(
        family_results,
        trusted_detection_counts
    )

    family_gate_ok, family_gate_reason = family_confidence_passes(
        family_results,
        family_decision_summary["forced_family"]
    )

    if not family_results:
        return {
            "top_tag": "General",
            "top_tags": ["General"],
            "top_family": "Vibes",
            "family_gate_ok": False,
            "family_gate_reason": family_gate_reason,
            "family_results": family_results,
            "canonical_results": [],
            "all_detection_counts": dict(all_detection_counts),
            "trusted_detection_counts": dict(trusted_detection_counts),
            "context": ctx,
        }

    top_family = family_results[0][0]

    selected_tag_map = {}
    for tag, spec in TAG_SPECS.items():
        if spec["family"] == top_family:
            selected_tag_map[tag] = spec["prompts"]

    tag_prompts, tag_keys = flatten_nested_prompt_map(selected_tag_map)
    tag_scores, _ = run_vlm(
        processor=processor,
        model=vlm,
        device=device,
        image_path=image_path,
        all_prompts=tag_prompts
    )

    collapsed_tags = collapse_scores(
        keys=tag_keys,
        scores=tag_scores,
        prompts=tag_prompts
    )
    canonical_results = sort_collapsed(collapsed_tags)
    canonical_results, traced_tag_results, tag_decision_summary = apply_tag_fusion(
        canonical_results,
        trusted_detection_counts
    )

    top_tags = [tag for tag, _, _ in canonical_results[:top_k_tags]]

    return {
        "top_tag": top_tags[0] if top_tags else "None",
        "top_tags": top_tags,
        "top_family": top_family,
        "family_gate_ok": family_gate_ok,
        "family_gate_reason": family_gate_reason,
        "family_results": family_results,
        "canonical_results": canonical_results,
        "all_detection_counts": dict(all_detection_counts),
        "trusted_detection_counts": dict(trusted_detection_counts),
        "context": ctx,
        "traced_family_results": traced_family_results,
        "family_decision_summary": family_decision_summary,
        "traced_tag_results": traced_tag_results,
        "tag_decision_summary": tag_decision_summary,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Working directory:", os.getcwd())
    print("Image path:", os.path.abspath(IMAGE_PATH))

    validate_image_for_pipeline(IMAGE_PATH)

    pipeline = get_pipeline()
    device = pipeline["device"]
    processor = pipeline["processor"]
    vlm = pipeline["vlm"]
    detector = pipeline["detector"]

    detections, detect_time = run_yolo_detection(detector, IMAGE_PATH)
    all_detection_counts = summarize_detections(detections)
    trusted_detection_counts, _ = summarize_trusted_detections(detections)
    print_detections(all_detection_counts, trusted_detection_counts, detect_time)

    family_prompt_map = {family: spec["prompts"] for family, spec in FAMILY_SPECS.items()}
    family_prompts, family_keys = flatten_simple_prompt_map(family_prompt_map)
    family_scores, family_time = run_vlm(
        processor=processor,
        model=vlm,
        device=device,
        image_path=IMAGE_PATH,
        all_prompts=family_prompts
    )

    collapsed_families = collapse_scores(
        keys=family_keys,
        scores=family_scores,
        prompts=family_prompts
    )
    family_results = sort_collapsed(collapsed_families)
    family_results, traced_family_results, ctx, decision_summary = apply_family_fusion(
        family_results,
        trusted_detection_counts
    )

    top_families = [family for family, _, _ in family_results[:TOP_FAMILIES_TO_KEEP]]
    family_gate_ok, family_gate_reason = family_confidence_passes(
        family_results,
        decision_summary["forced_family"]
    )

    print_family_stage(
        family_results=family_results,
        top_families=top_families,
        stage_time=family_time,
        family_gate_ok=family_gate_ok,
        family_gate_reason=family_gate_reason
    )
    print_context_signals(ctx)
    print_family_evidence(traced_family_results, decision_summary)

    if not family_gate_ok:
        print_no_strong_match(CLAIMED_TAGS)
        return

    selected_tag_map = {}
    for tag, spec in TAG_SPECS.items():
        if spec["family"] in top_families:
            selected_tag_map[tag] = spec["prompts"]

    tag_prompts, tag_keys = flatten_nested_prompt_map(selected_tag_map)
    tag_scores, tag_time = run_vlm(
        processor=processor,
        model=vlm,
        device=device,
        image_path=IMAGE_PATH,
        all_prompts=tag_prompts
    )

    collapsed_tags = collapse_scores(
        keys=tag_keys,
        scores=tag_scores,
        prompts=tag_prompts
    )
    canonical_results = sort_collapsed(collapsed_tags)
    canonical_results, traced_tag_results, tag_decision_summary = apply_tag_fusion(
        canonical_results,
        trusted_detection_counts
    )

    top_family = top_families[0]
    direct_supported, family_supported, unsupported, primary_inferred, bonus_inferred = build_verifier_summary(
        canonical_results=canonical_results,
        claimed_tags=CLAIMED_TAGS,
        top_family=top_family
    )

    rewards = resolve_rewards(
        direct_supported=direct_supported,
        family_supported=family_supported,
        primary_inferred=primary_inferred,
        bonus_inferred=bonus_inferred
    )

    print_tag_stage(
        canonical_results=canonical_results,
        claimed_tags=CLAIMED_TAGS,
        direct_supported=direct_supported,
        family_supported=family_supported,
        unsupported=unsupported,
        primary_inferred=primary_inferred,
        bonus_inferred=bonus_inferred,
        rewards=rewards,
        stage_time=tag_time
    )
    print_tag_evidence(
        traced_tag_results=traced_tag_results,
        tag_decision_summary=tag_decision_summary,
        limit=TOP_K_CANONICAL
    )


if __name__ == "__main__":
    main()