import albumentations as A
import cv2


def get_train_augmentations(img_size=224):
    return A.Compose([
        # 1️⃣ Scale image so longest side = img_size
        # Preserves aspect ratio
        A.LongestMaxSize(max_size=img_size),

        # 2️⃣ Pad to square (normalizes framing)
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0
        ),

        # 3️⃣ Mild geometry (safe)
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-30, 30),
            p=0.5
        ),
        A.Rotate(limit=30, p=0.2),

        # 4️⃣ Very mild photometric augmentation
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=1.0
            ),
            A.CLAHE(
                clip_limit=2,
                tile_grid_size=(8, 8),
                p=1.0
            ),
        ], p=0.2),
        A.ToGray(p=0.2),

        # 5️⃣ Rare degradation (optional, light)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(std_range=(0.0, 0.03), p=1.0),
            A.ImageCompression(quality_range=(75, 95), p=1.0),
        ], p=0.15),
    ])

def get_val_augmentations(img_size=224):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0
        ),
    ])