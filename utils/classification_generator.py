import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from math import ceil

class ClassificationDataset(tf.keras.utils.Sequence):
    """
    On-the-fly dataset for image classification with Albumentations augmentations.
    Works for training, validation, and prediction.
    """

    def __init__(self, root_dir, batch_size=16, augmentations=None, shuffle=True, num_classes=6):
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.augmentations = augmentations  # Albumentations Compose or None
        self.shuffle = shuffle
        self.num_classes = num_classes

        # Collect all samples
        self.samples = list(self.root_dir.glob("*/*"))

        # Map class folder name to integer label
        self.class_names = sorted({p.parent.name for p in self.samples})
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}

        self.on_epoch_end()

    def __len__(self):
        """Return number of batches per epoch (ceil to include last partial batch)"""
        return ceil(len(self.samples) / self.batch_size)

    def on_epoch_end(self):
        """Shuffle samples at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __getitem__(self, idx):
        """Generate one batch of data"""
        # Compute batch indices
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_samples = self.samples[start_idx:end_idx]

        images = []
        labels = []

        # Handle empty batch (should not happen with __len__ corrected)
        if len(batch_samples) == 0:
            raise IndexError(f"Batch {idx} is empty. Check __len__ implementation.")

        for path in batch_samples:
            # Load image
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply augmentations if provided
            if self.augmentations is not None:
                img = self.augmentations(image=img)["image"]

            # Normalize for MobileNetV2
            img = preprocess_input(img.astype(np.float32))

            images.append(img)

            # One-hot label
            label_idx = self.class_map[path.parent.name]
            label_one_hot = tf.keras.utils.to_categorical(label_idx, num_classes=self.num_classes)
            labels.append(label_one_hot)

        # Convert to NumPy arrays
        return np.array(images), np.array(labels)