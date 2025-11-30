#!/usr/bin/env python3
"""
Stable Diffusion 2.1 - Maximum specificity prompts
FINAL ATTEMPT before switching to augmentation
"""
import torch
from diffusers import StableDiffusionPipeline
import os
from tqdm import tqdm
import argparse

def setup_pipeline():
    model_id = "stabilityai/stable-diffusion-2-1"
    print("Loading Stable Diffusion 2.1...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    print(f"✅ Pipeline loaded on {device}")
    return pipe

# MAXIMUM SPECIFICITY - Every visual detail described
PROMPTS = {
    'no_leak': [
        "close-up photograph of dry gray metal industrial pipe with valve, completely dry surface texture, no moisture visible anywhere, matte gray metal, concrete floor visible below pipe, factory warehouse setting, normal fluorescent lighting, no water drops, no oil stains, no wet areas, realistic photograph",
        
        "industrial metal piping system, silver gray pipes, completely dry metal surface, no liquids present, factory floor concrete visible, overhead factory lights, pipe joints visible, no dripping, no wet spots, no moisture, maintenance inspection photograph, realistic",
        
        "dry industrial valve on gray metal pipe, close-up view, rough metal texture visible, no fluids anywhere, factory concrete floor in background, standard industrial lighting, no leaks present, no wet surface, no oil, no water, documentary photograph",
        
        "gray industrial pipe close-up, dry metal surface, valve connection visible, no moisture on pipe, no puddles on floor, concrete factory floor, fluorescent warehouse lighting, no liquids dripping, no stains, realistic industrial photograph",
        
        "industrial piping close-up photograph, gray metal pipes, completely dry, no water droplets, no oil residue, valve fittings visible, concrete floor beneath, factory lighting, no leaking, no wet areas, maintenance documentation photo",
    ],
    
    'oil_leak': [
        "close-up photograph showing thick black viscous liquid dripping down from crack in gray metal pipe, dark oil drops falling vertically downward, black puddle forming on concrete floor below pipe, oil stain visible on metal surface, factory fluorescent lighting, realistic photograph of industrial leak",
        
        "industrial gray metal pipe with visible crack, thick black oil liquid actively dripping out of crack, dark viscous fluid running down pipe surface vertically, black oil puddle on concrete floor beneath leak, oil drops mid-air falling, factory setting, realistic documentation photograph",
        
        "gray industrial pipe joint leaking thick black oil, dark viscous liquid flowing downward on metal surface, black oil drops dripping from pipe onto concrete floor, oil puddle visible below, black liquid stain on pipe, warehouse lighting, realistic leak detection photograph",
        
        "close-up of industrial pipe leak, thick black oil dripping from damaged section, dark viscous liquid running down gray metal pipe, oil drops falling onto floor, black puddle forming on concrete, oil stain visible, factory environment, realistic industrial photograph",
        
        "gray metal pipe with oil leak, thick black viscous liquid actively dripping from crack, dark oil flowing down pipe surface, black drops falling vertically, oil puddle on concrete floor below, dark stain on metal, fluorescent lighting, realistic maintenance photograph",
    ],
    
    'water_leak': [
        "close-up photograph of clear transparent water dripping from gray metal industrial pipe, water drops falling vertically downward, wet concrete floor visible below pipe, water puddle forming, transparent liquid visible on metal surface, factory fluorescent lighting, realistic photograph of water leak",
        
        "industrial gray metal pipe with water leak, clear transparent liquid actively dripping from crack, water running down pipe surface vertically, transparent water drops mid-air falling, wet spot on concrete floor beneath, water puddle visible, factory setting, realistic documentation photograph",
        
        "gray industrial pipe joint leaking clear water, transparent liquid flowing downward on metal surface, water drops dripping from pipe onto concrete floor, wet area visible below, clear liquid on pipe, warehouse lighting, realistic leak detection photograph",
        
        "close-up of pipe water leak, clear transparent water dripping from damaged pipe section, water running down gray metal surface, transparent drops falling onto floor, wet concrete visible, water puddle forming below, factory environment, realistic industrial photograph",
        
        "gray metal pipe leaking clear water, transparent liquid actively dripping from crack in pipe, water flowing down metal surface, clear drops falling vertically, wet concrete floor below with water puddle, transparent liquid visible, fluorescent lighting, realistic maintenance photograph",
    ]
}

# MAXIMUM negative prompt - everything we DON'T want
NEGATIVE_PROMPT = "artistic, painting, drawing, illustration, sketch, digital art, render, CGI, 3D render, cartoon, anime, manga, stylized, abstract, surreal, fantasy, science fiction, person, people, human, worker, man, woman, face, hands, body, employee, technician, bokeh, depth of field, blurred background, dramatic lighting, sunset, golden hour, sunrise, studio lighting, professional photography, beauty shot, glamour, fashion, colorful, vibrant, saturated colors, neon, glowing, shiny, polished, perfect, clean composition, symmetrical, centered, art photography, stock photo, advertisement, commercial, promotional"

def generate_images(pipe, class_name, num_images, output_dir, experiment):
    os.makedirs(output_dir, exist_ok=True)
    prompts = PROMPTS[class_name]
    
    print(f"\n{'='*60}")
    print(f"Generating {num_images} {class_name} images")
    print(f"Maximum specificity prompts + strong negative prompt")
    print(f"{'='*60}\n")
    
    generated = 0
    prompt_idx = 0
    
    with tqdm(total=num_images) as pbar:
        while generated < num_images:
            prompt = prompts[prompt_idx % len(prompts)]
            
            image = pipe(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=25,  # Increased from 25
                guidance_scale=9.0,      # Increased from 7.5
                height=512,
                width=512
            ).images[0]
            
            filename = f"{class_name}_{experiment}_synthetic_{generated:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            
            generated += 1
            prompt_idx += 1
            pbar.update(1)
    
    print(f"✅ Generated {generated} images")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['b', 'c'], required=True)
    parser.add_argument('--class_name', choices=['no_leak', 'oil_leak', 'water_leak'], required=True)
    parser.add_argument('--num_images', type=int, required=True)
    
    args = parser.parse_args()
    output_dir = f"synthetic_generation/experiment_{args.experiment}/{args.class_name}"
    
    pipe = setup_pipeline()
    generate_images(pipe, args.class_name, args.num_images, output_dir, f"exp_{args.experiment}")
    
    print(f"\n🎉 Complete! Images in: {output_dir}")

if __name__ == "__main__":
    main()
