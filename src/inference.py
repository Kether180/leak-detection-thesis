import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from datetime import datetime
import csv
import cv2
import numpy as np

# ---- CONFIG ----
IMAGES_FOLDER = "inference_images"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "leak_detector_best.pt")

# Create folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated_inference")
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# ---- TRANSFORMS (same as validation) ----
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- DEVICE ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running inference on GPU:", torch.cuda.is_available())

# ---- LOAD MODEL ----
classes = ["oil_leak", "water_leak"]     # <- IMPORTANT
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---- CSV OUTPUT ----
csv_path = os.path.join(OUTPUT_DIR, "inference_results.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["image", "predicted_class", "confidence (%)"])

# ---- INFERENCE ----
print("Images folder:", IMAGES_FOLDER)

for img_name in sorted(os.listdir(IMAGES_FOLDER)):
    img_path = os.path.join(IMAGES_FOLDER, img_name)
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        print(f"Skipping invalid file: {img_name}")
        continue

    tensor = val_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)
        predicted = classes[pred_idx.item()]
        confidence = confidence.item() * 100

    # Print to console
    print(f"{img_name:<25} → {predicted:<10} ({confidence:.2f}%)")

    # Save CSV
    csv_writer.writerow([img_name, predicted, f"{confidence:.2f}"])

    # ---- Annotated image ----
    img_cv = cv2.imread(img_path)
    if img_cv is not None:
        label = f"{predicted} ({confidence:.1f}%)"
        cv2.putText(img_cv, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)
        out_path = os.path.join(ANNOTATED_DIR, img_name)
        cv2.imwrite(out_path, img_cv)

csv_file.close()
print("\nInference complete — results saved to:")
print(f" • {csv_path}")
print(f" • Annotated images → {ANNOTATED_DIR}")

