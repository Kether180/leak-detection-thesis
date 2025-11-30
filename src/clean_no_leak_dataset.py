import os
import cv2
import imagehash
from PIL import Image
import numpy as np
from tqdm import tqdm

INPUT_DIR = "raw_no_leak_to_clean"
OUTPUT_DIR = "cleaned_no_leak_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_RESOLUTION = 400           # minimum height or width
DARK_THRESHOLD = 40            # too dark
BRIGHT_THRESHOLD = 230         # too bright

seen_hashes = set()

for fname in tqdm(os.listdir(INPUT_DIR)):
    fpath = os.path.join(INPUT_DIR, fname)

    try:
        img = Image.open(fpath).convert("RGB")
    except:
        continue

    # 1) resolution filter
    w, h = img.size
    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        continue

    # 2) duplicate filter
    hsh = imagehash.average_hash(img)
    if hsh in seen_hashes:
        continue
    seen_hashes.add(hsh)

    # 3) dark / bright filter
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    mean_val = gray.mean()
    if mean_val < DARK_THRESHOLD or mean_val > BRIGHT_THRESHOLD:
        continue

    # 4) save cleaned image as jpg
    out_name = os.path.splitext(fname)[0] + ".jpg"
    img.save(os.path.join(OUTPUT_DIR, out_name), "JPEG", quality=95)

print("\n Cleaning complete!")
print("Cleaned images saved to:", OUTPUT_DIR)
print("Total kept:", len(os.listdir(OUTPUT_DIR)))

