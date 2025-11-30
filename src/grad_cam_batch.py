import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import cv2   # ✅ Added for proper infrared heatmap

MODEL_PATH = "outputs/leak_detector_best.pt"
TEST_DIR = "dataset/test"
OUT_DIR = "outputs/gradcam_batch"

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# model
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, 2)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# hooks
target_layer = model.layer4[-1].conv2
feats, grads = [], []

def fwd_hook(m, i, o):
    feats.append(o)

def bwd_hook(m, gi, go):
    grads.append(go[0])

target_layer.register_forward_hook(fwd_hook)
target_layer.register_full_backward_hook(bwd_hook)


# ✅ REPLACED FUNCTION (proper infrared Grad-CAM)
def save_cam_overlay(original_img, cam, out_path):
    """
    Save Grad-CAM using infrared style (COLORMAP_JET) and overlay on original image.
    """

    # Normalize CAM
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = (cam * 255).astype(np.uint8)

    # Resize CAM to match original resolution
    cam = cv2.resize(cam, (original_img.width, original_img.height))

    # Apply infrared color map (thermal camera style)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # Convert PIL → OpenCV (RGB → BGR)
    original_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)

    # Blend 60% original + 40% infrared heatmap
    blended = cv2.addWeighted(original_cv, 0.6, heatmap, 0.4, 0)

    # Convert back to RGB for saving
    blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    # Save
    Image.fromarray(blended).save(out_path)



def gradcam_single(path):
    feats.clear()
    grads.clear()

    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    # forward
    scores = model(x)
    pred = scores.argmax().item()

    # backward
    model.zero_grad()
    scores[0, pred].backward()

    # get activations & grads
    A = feats[-1].to(device)     # [1, C, H, W]
    G = grads[-1].to(device)     # [C, H, W]

    # weights = mean gradient per channel
    weights = G.mean(dim=(1,2))

    # CAM
    cam = torch.zeros(A.shape[2:], device=device)
    for c, w in enumerate(weights):
        cam += w * A[0, c]

    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    # Convert to numpy CPU
    cam_np = cam.detach().cpu().numpy()

    # output name
    out_name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUT_DIR, f"gradcam_{out_name}.png")

    # ✅ Use infrared overlay function
    save_cam_overlay(img, cam_np, out_path)

    print(" Saved →", out_path)



# run batch
for cls in ["oil", "water"]:
    folder = os.path.join(TEST_DIR, cls)
    for name in os.listdir(folder):
        if name.lower().endswith((".jpg",".jpeg",".png")):
            gradcam_single(os.path.join(folder, name))

print("\n Batch Grad-CAM complete!")
print(f"Saved in: {OUT_DIR}")
