import random
import shutil
from pathlib import Path
from collections import defaultdict

def split_classification_dataset(
    raw_dir,
    output_dir,
    ratios=(0.6, 0.2, 0.2),  # Train / Val / Test split ratios
    seed=42
):
    random.seed(seed)  # For reproducibility

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    total_counts = {"train": 0, "val": 0, "test": 0}

    # Function to split a list of images into train/val/test
    def split_group(imgs):
        n = len(imgs)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        return {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:]
        }

    for class_dir in raw_dir.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*"))

        # Group images by type
        groups = {
            "onlyhand": [img for img in images if "onlyhand" in img.stem],
            "selfie":   [img for img in images if "selfie" in img.stem],
            "photo":    [img for img in images if "photo" in img.stem],  # Friend's dataset
        }

        # Check the old 50/50 balance between onlyhand/selfie
        if groups["onlyhand"] and groups["selfie"]:
            assert len(groups["onlyhand"]) == len(groups["selfie"]), \
                f"Class {class_dir.name} is not 50/50 onlyhand/selfie"

        # Dictionary to store final splits for this class
        class_splits = defaultdict(list)

        # Split each group independently
        for group_name, imgs in groups.items():
            if not imgs:
                continue

            random.shuffle(imgs)  # Shuffle before splitting
            group_split = split_group(imgs)

            # Add images to the corresponding split
            for split in ["train", "val", "test"]:
                class_splits[split].extend(group_split[split])

        # Shuffle each split to mix different image types
        for split in class_splits:
            random.shuffle(class_splits[split])

        # Copy images to output directories
        for split_name, imgs in class_splits.items():
            out_dir = output_dir / split_name / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for img in imgs:
                shutil.copy(img, out_dir / img.name)

            total_counts[split_name] += len(imgs)

        # Print class-level summary
        print(
            f"Class {class_dir.name}: "
            f"train={len(class_splits['train'])}, "
            f"val={len(class_splits['val'])}, "
            f"test={len(class_splits['test'])} | "
            f"onlyhand={len(groups['onlyhand'])}, "
            f"selfie={len(groups['selfie'])}, "
            f"photo={len(groups['photo'])}"
        )

    # Print overall dataset summary
    print("\nDataset split completed:")
    print(f"Train images: {total_counts['train']}")
    print(f"Val images:   {total_counts['val']}")
    print(f"Test images:  {total_counts['test']}")
    print(f"Total:        {sum(total_counts.values())}")
