import os
import time
from collections import Counter

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .config import TRUSTED_DETECTOR_CLASSES


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


def load_yolo(model_name: str):
    print("Loading YOLO detector...")
    return YOLO(model_name)


def run_yolo_detection(detector, image_path: str):
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