import os
import cv2
from PIL import Image
from tqdm import tqdm

# Path to dataset folder
DATASET_DIR = "leaks_dataset"  # folders /oil_leak /water_leak (each with /real /synthetic)
TARGET_SIZE = (224, 224)  # suitable for ResNet18 / ResNet50

# === HELPER FUNCTIONS ===
def is_blurry(img, threshold=100.0):
    """Check if image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def clean_and_resize_images(root_dir):
    total_processed = 0
    total_removed = 0

    for subdir, _, files in os.walk(root_dir):
        for file in tqdm(files, desc=f"Cleaning {subdir}"):
            path = os.path.join(subdir, file)

            # Skip non-image files
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    os.remove(path)
                    total_removed += 1
                except:
                    pass
                continue

            # Try reading image
            img = cv2.imread(path)
            if img is None:
                os.remove(path)
                total_removed += 1
                continue

            # Remove blurry images
            if is_blurry(img):
                os.remove(path)
                total_removed += 1
                continue

            # Resize and convert color space
            img = cv2.resize(img, TARGET_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img).save(path)
            total_processed += 1

    print(f" Cleaned {total_processed} images. Removed {total_removed} invalid or low-quality ones.")

# === MAIN ===
if __name__ == "__main__":
    clean_and_resize_images(DATASET_DIR)

