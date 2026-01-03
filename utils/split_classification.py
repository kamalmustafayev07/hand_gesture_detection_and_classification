import random
import shutil
from pathlib import Path

def split_classification_dataset(
    raw_dir,
    output_dir,
    ratios=(0.6, 0.2, 0.2),  # Train/val/test split ratios
    seed=42
):
    random.seed(seed)  # Ensure reproducibility

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    total_counts = {"train": 0, "val": 0, "test": 0}  # Count of images per split

    for class_dir in raw_dir.iterdir():  # Iterate over each class folder
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*"))
        # Separate images by type
        onlyhand_imgs = [img for img in images if "onlyhand" in img.stem]
        selfie_imgs   = [img for img in images if "selfie" in img.stem]

        # Check initial balance (50/50)
        assert len(onlyhand_imgs) == len(selfie_imgs), \
            f"Class {class_dir.name} is not 50/50 initially"

        random.shuffle(onlyhand_imgs)  # Shuffle before splitting
        random.shuffle(selfie_imgs)

        # Function to split one group into train/val/test
        def split_group(imgs):
            n = len(imgs)
            n_train = int(n * ratios[0])
            n_val   = int(n * ratios[1])
            return {
                "train": imgs[:n_train],
                "val": imgs[n_train:n_train + n_val],
                "test": imgs[n_train + n_val:]
            }

        onlyhand_split = split_group(onlyhand_imgs)
        selfie_split   = split_group(selfie_imgs)

        # Combine splits from both types
        splits = {
            split: onlyhand_split[split] + selfie_split[split]
            for split in ["train", "val", "test"]
        }

        # Shuffle each split
        for split in splits:
            random.shuffle(splits[split])

        # Copy images to output folders
        for split_name, imgs in splits.items():
            out_dir = output_dir / split_name / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for img in imgs:
                shutil.copy(img, out_dir / img.name)

            total_counts[split_name] += len(imgs)

        # Print class-level summary
        print(
            f"Class {class_dir.name}: "
            f"train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])} | "
            f"onlyhand={len(onlyhand_imgs)}, selfie={len(selfie_imgs)}"
        )

    # Print final summary
    print("\nDataset split completed:")
    print(f"Train images: {total_counts['train']}")
    print(f"Val images:   {total_counts['val']}")
    print(f"Test images:  {total_counts['test']}")
    print(f"Total:        {sum(total_counts.values())}")
