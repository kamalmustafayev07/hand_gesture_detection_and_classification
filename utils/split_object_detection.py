from pathlib import Path
import shutil
import random

#Function for splitting the object detection data into train, validation, test sets
def split_od_dataset(
    images_dir,
    labels_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    seed=42  # for reproducibility
):
    random.seed(seed) # Ensure reproducibility
    
    # Convert paths to Path objects for easier handling
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    # Get all JPG images from the images directory
    all_images = list(images_dir.glob("*.jpg"))
    
    # Separate images into two groups based on filename
    onlyhand_images = [img for img in all_images if "onlyhand" in img.stem]
    selfie_images = [img for img in all_images if "selfie" in img.stem]
    
    # Print counts for verification
    print(f"Found onlyhand images: {len(onlyhand_images)}")
    print(f"Found selfie images: {len(selfie_images)}")
    
    # Shuffle each group independently
    random.shuffle(onlyhand_images)
    random.shuffle(selfie_images)
    
    # Calculate split sizes for each class
    n_onlyhand = len(onlyhand_images)
    n_selfie = len(selfie_images)
    
    n_train_onlyhand = int(n_onlyhand * train_ratio)
    n_val_onlyhand = int(n_onlyhand * val_ratio)
    
    n_train_selfie = int(n_selfie * train_ratio)
    n_val_selfie = int(n_selfie * val_ratio)
    
    # Create lists for each split by taking proportional parts from each class
    train_imgs = onlyhand_images[:n_train_onlyhand] + selfie_images[:n_train_selfie]
    val_imgs = onlyhand_images[n_train_onlyhand:n_train_onlyhand + n_val_onlyhand] + \
               selfie_images[n_train_selfie:n_train_selfie + n_val_selfie]
    test_imgs = onlyhand_images[n_train_onlyhand + n_val_onlyhand:] + \
                selfie_images[n_train_selfie + n_val_selfie:]
    
    # Shuffle within each split to avoid blocks of the same class
    random.shuffle(train_imgs)
    random.shuffle(val_imgs)
    random.shuffle(test_imgs)
    
    # Define splits dictionary
    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }
    
    # Copy images and corresponding labels to the output directories
    for split_name, imgs in splits.items():
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img in imgs:
            label_path = labels_dir / (img.stem + ".txt")
            
            # Copy image
            shutil.copy(img, img_dir / img.name)
            
            # Copy label if it exists
            if label_path.exists():
                shutil.copy(label_path, lbl_dir / label_path.name)
            else:
                print(f"Warning: Label not found for {img.name}")
    
    # Print final distribution for verification
    print("\nDistribution after splitting:")
    for split_name, imgs in splits.items():
        onlyhand_count = sum(1 for img in imgs if "onlyhand" in img.stem)
        selfie_count = len(imgs) - onlyhand_count
        print(f"{split_name}: onlyhand={onlyhand_count}, selfie={selfie_count} (total {len(imgs)})")