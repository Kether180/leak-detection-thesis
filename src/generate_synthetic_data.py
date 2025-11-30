import torch
import random
from datetime import datetime
import os
import argparse
from diffusers import StableDiffusionPipeline


# Base prompts

BASE_PROMPTS = {
    "oil": (
        "close-up photo of amber industrial oil leaking from a cracked hydraulic hose "
        "attached to a robotic actuator in a factory environment, photorealistic, "
        "high detail, industrial machinery, metal textures, reflections"
    ),
    "water": (
        "close-up photo of clear water leaking or dripping from a damaged pipe joint "
        "inside a factory or robotic environment, puddles forming, realistic lighting, "
        "wet surfaces, photorealistic"
    ),
}


# Variation sets

LIGHT_VARIANTS = [
    "harsh industrial lighting",
    "soft diffused lighting",
    "cold bluish lighting",
    "warm yellow lighting",
    "low light conditions",
]

CAMERA_VARIANTS = [
    "macro close-up",
    "slightly tilted angle",
    "top-down view",
    "side angle perspective",
    "shallow depth of field",
]

ENV_VARIANTS = [
    "robotic arm components visible",
    "pipes and cables in background",
    "metallic surfaces",
    "industrial factory background",
    "wet concrete floor reflections",
]

OIL_BEHAVIOR = [
    "thick viscous oil dripping slowly",
    "thin oily streaks running down metal",
    "droplets forming and stretching",
    "oil pooling on the ground",
]

WATER_BEHAVIOR = [
    "steady dripping forming ripples",
    "fast leak spraying outward slightly",
    "water spreading across the floor",
    "small droplets falling mid-air",
]


# Build dynamic prompt

def build_prompt(leak_type: str) -> str:
    base = BASE_PROMPTS[leak_type]
    behavior = random.choice(OIL_BEHAVIOR if leak_type == "oil" else WATER_BEHAVIOR)

    final = (
        f"{base}, {behavior}, "
        f"{random.choice(LIGHT_VARIANTS)}, "
        f"{random.choice(CAMERA_VARIANTS)}, "
        f"{random.choice(ENV_VARIANTS)}, "
        "ultra-photorealistic, 4K, high detail"
    )
    return final



# Main function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--leak_type", type=str, required=True, choices=["oil", "water"])
    args = parser.parse_args()

    leak_type = args.leak_type
    num_images = args.num_images

    # Folder mapping (IMPORTANT)
    FOLDER_MAP = {"oil": "oil_leak", "water": "water_leak"}
    folder = FOLDER_MAP[leak_type]

    # Absolute dataset path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "leaks_dataset", folder, "synthetic")
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading Stable Diffusion 2.1 in FP32 mode (GTX 1080Ti safe)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float32,
        safety_checker=None,
        feature_extractor=None,
    ).to("cuda")

    print(f"[INFO] Generating {num_images} synthetic images for: {leak_type}")

    for i in range(num_images):
        prompt = build_prompt(leak_type)
        print(f"  → {i+1}/{num_images}: {prompt}")

        image = pipe(prompt).images[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{leak_type}_synthetic_{i}_{timestamp}.png"
        image.save(filename)

        print(f"      Saved: {filename}")

    print("\n[DONE] Synthetic data generation completed.\n")


if __name__ == "__main__":
    main()

