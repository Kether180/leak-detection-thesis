import os
import shutil
import random
import argparse

def split_dataset(input_dir, output_dir, train_ratio, val_ratio, test_ratio):
    os.makedirs(output_dir, exist_ok=True)

    # Create output subfolders
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Go through each class (Oil, Water)
    for category in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, category)
        if not os.path.isdir(class_dir):
            continue

        # Gather all images
        all_images = []
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(root, file))

        random.shuffle(all_images)
        n_total = len(all_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        splits_dict = {
            'train': all_images[:n_train],
            'val': all_images[n_train:n_train + n_val],
            'test': all_images[n_train + n_val:]
        }

        for split, files in splits_dict.items():
            split_class_dir = os.path.join(output_dir, split, category)
            os.makedirs(split_class_dir, exist_ok=True)
            for f in files:
                shutil.copy(f, split_class_dir)

    print(" Dataset successfully split into train/val/test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test sets")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="dataset")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    split_dataset(args.input, args.output, args.train, args.val, args.test)

