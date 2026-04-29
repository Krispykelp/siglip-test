import time

import torch
from transformers import AutoModel, AutoProcessor


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


def load_vlm(model_name, device):
    print("Loading VLM...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


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


def run_vlm(processor, model, device, image_path, all_prompts, load_pil_image_fn):
    image = load_pil_image_fn(image_path)
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