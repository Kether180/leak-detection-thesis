import os
from PIL import Image

ROOT = "dataset"

bad = []

for split in ["train", "val", "test"]:
    for cls in os.listdir(os.path.join(ROOT, split)):
        folder = os.path.join(ROOT, split, cls)
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            try:
                img = Image.open(path)
                img.verify()       # detects truncated images
                img = img.convert("RGB")  # detects wrong channels
            except Exception as e:
                bad.append(path)
                print(f"[BAD] {path}  -> {e}")

print("\nTotal bad images:", len(bad))
if bad:
    print("\nDelete them with:")
    for b in bad:
        print(f'rm "{b}"')

