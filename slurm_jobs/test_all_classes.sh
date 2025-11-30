#!/bin/bash
# Test all three classes with new prompts

echo "🧪 Testing all three classes (3 images each)"
echo "============================================"

# Clean old test images
rm -rf synthetic_generation/experiment_b/*/

# Generate 3 of each
echo -e "\n1️⃣ Generating NO-LEAK images..."
python generate_synthetic_images.py --experiment b --class_name no_leak --num_images 3

echo -e "\n2️⃣ Generating OIL-LEAK images..."
python generate_synthetic_images.py --experiment b --class_name oil_leak --num_images 3

echo -e "\n3️⃣ Generating WATER-LEAK images..."
python generate_synthetic_images.py --experiment b --class_name water_leak --num_images 3

echo -e "\n✅ Test complete! Check images:"
echo "   synthetic_generation/experiment_b/no_leak/"
echo "   synthetic_generation/experiment_b/oil_leak/"
echo "   synthetic_generation/experiment_b/water_leak/"
