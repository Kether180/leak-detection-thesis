#!/usr/bin/env python3
"""Generate supplement images (100 per class) to reach 4,500 total."""

import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import argparse
import random

PROMPTS = {
    "no_leak": [
        "industrial pipes and valves in factory, clean stainless steel surface, no leaks visible, no fluids, professional lighting, photorealistic industrial photography",
        "clean factory machinery close-up, industrial valve assembly, brushed metal finish, normal operation, no defects, no liquid visible, sharp focus",
        "pristine industrial piping system, metal pipes with flanges, factory setting, overhead lighting, no moisture, professional industrial photo",
        "well-maintained hydraulic system, clean metal surfaces, industrial facility, no leaks or drips, sharp detail, photorealistic",
        "factory floor with clean pipes and valves, stainless steel equipment, no fluids present, industrial environment, professional photography",
    ],
    "oil_leak": [
        "dark hydraulic oil puddle on concrete factory floor with rainbow sheen reflecting overhead fluorescent lighting, industrial manufacturing environment, photorealistic",
        "viscous amber oil leaking from hydraulic fitting, pooling on brushed steel platform with rainbow patterns, close-up industrial photography",
        "black oil dripping from industrial machinery onto factory floor, iridescent sheen visible, puddle forming, realistic industrial scene",
        "hydraulic oil leak from pipe joint, dark fluid with rainbow reflection on concrete, industrial setting, photorealistic detail",
        "motor oil spill on factory floor, dark viscous liquid with colorful sheen, industrial equipment in background, professional photography",
    ],
    "water_leak": [
        "burst pipe spraying water in industrial facility, wet concrete floor with clear water puddle forming, bright daylight through warehouse windows",
        "close-up of transparent water droplets leaking from cracked coolant line on machinery, wet metal surface with reflections, dim warehouse lighting",
        "water leak from industrial pipe, clear liquid pooling on factory floor, wet concrete surface, overhead fluorescent lighting, photorealistic",
        "dripping water from pipe fitting, transparent droplets on metal surface, industrial environment, realistic water reflections",
        "coolant water leak in factory, clear puddle on concrete floor, wet surfaces with reflections, industrial setting, professional photo",
    ],
}

NEGATIVE_PROMPT = "blurry, low quality, distorted, cartoon, anime, drawing, painting, sketch, artificial, fake looking, oversaturated, text, watermark, logo"

def generate_images(class_name, start_idx, end_idx, output_dir):
    print(f"Loading Stable Diffusion 2.1...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")
    print("✅ Pipeline loaded on cuda")
    
    output_path = Path(output_dir) / class_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    prompts = PROMPTS[class_name]
    
    print("=" * 60)
    print(f"Generating {class_name} images {start_idx} to {end_idx-1}")
    print("=" * 60)
    
    for idx in range(start_idx, end_idx):
        filename = f"{class_name}_exp_b_synthetic_{idx:04d}.jpg"
        filepath = output_path / filename
        
        if filepath.exists():
            print(f"⏭️  {filename} exists, skipping")
            continue
        
        prompt = random.choice(prompts)
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=50,
                guidance_scale=7.5,
            ).images[0]
        
        image.save(filepath, "JPEG", quality=95)
        print(f"✅ [{idx-start_idx+1}/{end_idx-start_idx}] Saved: {filename}")
    
    print(f"🎉 Complete! {class_name} images in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, required=True, choices=["no_leak", "oil_leak", "water_leak"])
    parser.add_argument("--start_idx", type=int, default=1400)
    parser.add_argument("--end_idx", type=int, default=1500)
    parser.add_argument("--output_dir", type=str, default="synthetic_generation/experiment_b")
    args = parser.parse_args()
    generate_images(args.class_name, args.start_idx, args.end_idx, args.output_dir)
