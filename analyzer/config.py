VLM_MODEL_NAME = "google/siglip2-base-patch16-224"
YOLO_MODEL_NAME = "yolo11n.pt"

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
DETECTOR_OVERRIDE_FAMILIES = {"Athletics", "Smarts", "Social", "Vibes"}

DIRECT_REWARD = 2
FAMILY_SUPPORT_REWARD = 1
PRIMARY_INFERRED_REWARD = 2
BONUS_REWARD = 1

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