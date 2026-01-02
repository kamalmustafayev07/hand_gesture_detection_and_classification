# Hand Gesture Recognition Project Documentation
## 1. Project Overview

This project implements a two-stage computer vision pipeline for hand gesture recognition, designed to identify numbers from 0 to 5 based on hand poses, including variations in finger configurations for classes 1 through 4 beyond just the index finger, while focusing on standard gestures as defined (e.g., closed fist for 0, all fingers up for 5). The system detects the hand region in an input image and classifies the gesture, addressing real-world challenges such as variability in lighting, backgrounds, hand orientations, shapes, skin tones, and limited data. The project emphasizes dataset collection—with assistance from friends for gathering images—annotation quality, model training, evaluation, and a reproducible inference pipeline, aligning with the assignment's focus on practical computer vision skills. The classification and cropping processes were optimized for the inner palm area, achieving strong performance despite data constraints: the best classification model reached 95% accuracy, and object detection attained mAP50 of 0.995 and mAP50-95 of 0.994.

### Objective
The primary goal is to recognize hand gestures representing numbers 0–5 using a multi-stage pipeline:
- **Object Detection**: Locate the hand region in the image, encompassing the area from the wrist to the fingertips.
- **Classification**: Identify the specific gesture from the cropped hand region, where:
  - 0: Closed fist
  - 1: Index finger up
  - 2: Two fingers up
  - 3: Three fingers up
  - 4: Four fingers up
  - 5: All five fingers up

This setup enables robust recognition while handling variations in hand shapes, skin tones, and environmental factors.

### Pipeline Summary
The end-to-end flow processes an input image as follows:
1. **Input Image**: A raw image containing a single visible hand gesture.
2. **Hand Detection**: The object detection model identifies the hand's bounding box.
3. **Cropping**: Extract the detected hand region from the image.
4. **Gesture Classification**: The classification model analyzes the cropped image to predict the gesture class (0–5).
5. **Output Prediction**: Display the original image with the bounding box overlaid and the predicted number.

This pipeline is implemented in a dedicated inference notebook, ensuring it works on unseen images, such as those provided by the instructor.

### Technologies Used
- **Object Detection**: Ultralytics YOLOv11 for efficient single-class hand detection, trained with custom configurations and data splits.
- **Classification**: TensorFlow/Keras with a fine-tuned MobileNetV2 backbone, augmented with custom head layers for multi-class prediction.
- **Image Processing and Augmentation**: OpenCV for handling images, cropping, and visualization; Albumentations library for data augmentations including scaling, padding, horizontal flips, affine transformations, rotations, brightness/contrast adjustments, CLAHE, grayscale conversion, and mild degradations like blur, noise, and compression to enhance robustness.
- **Data Management**: Custom Python utilities for dataset splitting, generators for batched data loading with normalization, and YOLO-compatible label formats.
- **Evaluation and Visualization**: Matplotlib and Seaborn for plotting loss curves, confusion matrices, and metrics; built-in YOLO tools for mAP, precision, and recall.

### Group Information
This is an individual project completed by one student. In accordance with the dataset sharing policy, the core dataset was collected and labeled independently. Additional data was shared from groupmate Elvin Mahmudzada to enhance the classification dataset, with clear documentation: the original dataset includes 444 images (222 "onlyhand" close-ups and 222 "selfie" full-view images, balanced at 74 per class), while Elvin's contribution added 18 images per class for improved variability and model robustness. All shared data was integrated post-labeling, and the project maintains full attribution for collected versus shared portions. 

## 2. Project Structure Overview

The project follows a **clear, modular, and reproducible folder structure**, separating configuration, data, models, experiments, results, and reusable utilities.

### Directory Layout

```text
project_root/
│
├── configuration/
│   └── detection.yaml
│
├── data/
│   ├── data.zip
│   └── data_added_elvin.zip
│
├── models/
│   ├── classification/
│   └── detection/
│
├── notebooks/
│   ├── 01-yolo_model_training.ipynb
│   ├── 02-classification_model_training.ipynb
│   └── 03-inference_pipeline.ipynb
│
├── results/
│   └── detection_metrics/
│       ├── training_hand_yolo11/
│       └── val_test_metrics/
│
└── utils/
    ├── classification_augmentations.py
    ├── classification_generator.py
    ├── classification_model_builder.py
    ├── inference.py
    ├── split_classification.py
    ├── split_classification_added_elvin.py
    └── split_object_detection.py
````

---

### Configuration

**`configuration/detection.yaml`**
Contains the full configuration for training the **YOLOv11 hand detection model**, including dataset paths, class definitions, and training parameters.

---

### Data

#### `data/data.zip`

The core dataset collected by the author, containing **444 images**:

* **222 `onlyhand` images** (close-up hand crops)
* **222 `selfie` images** (hand visible within a larger scene)

The dataset is balanced with **74 images per class (0–5)**, equally split between *onlyhand* and *selfie* images.

**Internal structure:**

* **`dataset_classification/`**

  * Six folders (`0`–`5`), one per class
  * Contains **cropped hand images**, obtained by extracting bounding boxes
* **`dataset_object_detection/`**

  * `images/`: original images
  * `labels/`: YOLO-HBB `.txt` annotation files with bounding box coordinates

Data annotation and cropping were performed using **X-AnyLabeling**.

---

#### `data/data_added_elvin.zip`

An extended dataset incorporating additional images from **Elvin Mahmudzada**:

* For **object detection**, the dataset structure mirrors the original dataset.
* For **classification**, **18 additional images per class** were added, improving variability and robustness.

These images are explicitly identifiable by filename patterns and are handled separately during dataset splitting to preserve class and source balance.

---

### Models

#### Detection (`models/detection/`)

* **`hand_detector_best.pt`** – best-performing YOLOv11 model
* **`hand_detector_last.pt`** – final model checkpoint after training completion

#### Classification (`models/classification/`)

* **`mobilenetv2_best.keras`** – first training run (~91% accuracy, ~0.54 loss)
* **`mobilenetv2_best_new.keras`** – later run with higher accuracy (~93%) but higher loss (~0.8)
* **`mobilenetv2_best_added_elvin.keras`** – final model trained with extended dataset, achieving **~95% accuracy** and **~0.39 loss**

---

### Notebooks

* **`01-yolo_model_training.ipynb`**
  Training and evaluation of the YOLOv11 hand detection model.

* **`02-classification_model_training.ipynb`**
  Training, fine-tuning, and evaluation of the MobileNetV2-based classification model.

* **`03-inference_pipeline.ipynb`**
  End-to-end inference pipeline combining detection, cropping, and classification on unseen images.

---

### Results

* **`results/detection_metrics/training_hand_yolo11/`**
  Saved training artifacts and logs from the YOLOv11 training stage.

* **`results/detection_metrics/val_test_metrics/`**
  Metrics and evaluation results from the **test set** (not validation), ensuring unbiased performance reporting.

---

### Utilities

The **`utils/`** directory contains reusable components imported across notebooks:

* **Augmentations** – training-time augmentations and validation-time resizing
* **Data generators** – batched data loading and MobileNetV2 normalization (`[-1, 1]`)
* **Model builder** – MobileNetV2 backbone with custom classification head
* **Dataset splitting**:

  * Balanced train/val/test splits for classification
  * Extended split logic preserving balance for Elvin’s added images
  * YOLO-compatible splits for object detection
* **Inference utilities** – shared logic for the complete detection + classification pipeline 

## 3. Dataset Description

This section describes the **data collection, annotation, organization, and sharing process** in accordance with the project guidelines. The dataset targets **hand gesture recognition for digits 0–5 shown with one hand**, allowing **natural variability in finger configurations** while enforcing a consistent constraint: the **inner side of the palm always faces the camera**.

The data is handled **separately for object detection and classification** to ensure proper supervision, balanced splits, and fair evaluation across both stages of the pipeline.

---

### Data Collection

- **Source**: All images were captured using a **smartphone camera**. No webcams or external imaging devices were used.
- **Participants**: The dataset was collected by the author with assistance from a friend, resulting in images of **multiple hands** to increase visual diversity.
- **Variability**: Images were intentionally collected under diverse conditions:
  - **Lighting**: indoor, outdoor, natural, artificial
  - **Backgrounds**: plain walls, cluttered indoor scenes, outdoor environments
  - **Hand orientation**: slight rotations and tilts while keeping the palm facing the camera
  - **Distance to camera**:
    - *onlyhand*: close-up images with the hand filling most of the frame
    - *selfie*: wider, contextual images where the hand appears within the full scene
- **Image Types Distribution**: A strict **50/50 split** between *onlyhand* and *selfie* images was maintained to improve real-world robustness.

**Dataset size (original):**
- **Total images**: 444  
  - 222 *onlyhand*
  - 222 *selfie*
- **Classes**: 6 (digits 0–5)
- **Per class**: 74 images (37 onlyhand + 37 selfie)

---

### Dataset Structure

All datasets are stored under the `data/` directory.

#### Object Detection Dataset  
**Path:** `data/data.zip → dataset_object_detection/`

```

dataset_object_detection/
├── images/
│   └── *.jpg
└── labels/
│   └── *.txt

```

- **Images**: 444 raw `.jpg` images
- **Labels**: YOLO-format `.txt` files with matching filenames
- **Classes**: Single class — `hand`
- **Label format**:
```

class_id x_center y_center width height

```
All values are **normalized to [0, 1]**.

---

#### Classification Dataset (Original)  
**Path:** `data/data.zip → dataset_classification/`

```

dataset_classification/
├── 0/
├── 1/
├── 2/
├── 3/
├── 4/
└── 5/

````

- **Content**: Cropped hand images extracted from detection bounding boxes
- **Per class**: 74 images
- **Purpose**: Gesture classification only

---

#### Classification Dataset (Augmented with Shared Data)  
**Path:** `data/data_added_elvin.zip → dataset_classification/`

- Builds upon the original classification dataset
- **Added data source**: Elvin Mahmudzada
- **Added images**: 18 per class (files prefixed with `photo`)
- **Total per class**: 92 images
  - 37 onlyhand
  - 37 selfie
  - 18 photo (shared)

This augmentation increases **inter-person variability**, hand shapes, and contextual diversity, improving generalization.

---

### Annotation Process

- **Tool**: **X-AnyLabeling**
- **Bounding boxes**:
  - Tightly enclose the hand region from **wrist to fingertips**
  - One bounding box per image
  - Single class: `hand`
- **Output format**:
  - YOLO-compatible `.txt` files
  - Stored under:
    ```
    dataset_object_detection/labels/
    ```

Bounding box annotations were also used to generate cropped images for the classification dataset.

---

### Data Sharing Policy

- **Independently collected data**:
  - All **444 images** in `data/data.zip` were collected and annotated solely by the author.
- **Shared data**:
  - **108 cropped images** (18 per class) from Elvin Mahmudzada
  - Used **only for classification**
  - No object detection labels were shared, reused, or modified

All shared data is clearly documented and traceable by filename conventions.

---

### Dataset Splitting

Dataset splits are handled using dedicated scripts in the `utils/` directory to ensure **reproducibility and balance**.

#### Object Detection Split  
**Script:** `utils/split_object_detection.py`

- **Default ratios**: 70% / 20% / 10% (train / val / test)
- **Preserves**:
  - `images/` and `labels/` directory structure
  - Balance between *onlyhand* and *selfie* images

**Output structure:**
```text
dataset_object_detection/
├── train/
├── val/
└── test/
```

#### Classification Split (Original Dataset)  
**Script:** `utils/split_classification.py`

- **Default ratios**: 60% / 20% / 20%
- **Guarantees**:
- Class balance across splits
- Equal proportion of *onlyhand* and *selfie* images per class

---

#### Classification Split (Augmented Dataset)  
**Script:** `utils/split_classification_added_elvin.py`

- Extends the original splitting logic
- Additionally balances:
- *onlyhand*
- *selfie*
- *photo* (shared data)
- Ensures proportional representation of all image sources across train, validation, and test sets

---

This structured approach ensures **transparent data provenance**, **balanced evaluation**, and **reproducible experiments** across both detection and classification stages. 

## 4. Data Augmentation

This section documents the **data augmentation strategy** applied during the training of the **classification model**, in line with the project guidelines. Augmentations were implemented using the **Albumentations** library to increase dataset diversity and address real-world challenges such as variations in lighting, hand orientation, camera distance, and image quality.

All augmentations are applied **on-the-fly during training**, meaning the raw dataset on disk remains unchanged. This approach improves model robustness while avoiding unnecessary data duplication.

---

### Augmentations Used  
**Source file:** `utils/classification_augmentations.py`  
**Training pipeline:** `get_train_augmentations`

The following transformations are applied dynamically during training:

#### Geometric Transformations
- **Resizing and Padding**  
  - The longest image side is scaled to **224 pixels** while preserving aspect ratio.
  - Images are padded to a **224 × 224** square with black borders.
  - This standardizes inputs for **MobileNetV2** while accommodating varying original dimensions.

- **Horizontal Flip**  
  - Probability: **50%**  
  - Simulates left/right hand symmetry.

- **Affine Transformations**  
  - Probability: **50%**
  - Includes:
    - Translation: ±10%
    - Scaling: 0.9–1.1
    - Rotation: ±30°
  - Mimics different camera distances and hand placements.

- **Additional Rotation**  
  - Probability: **20%**
  - Rotation range: ±30°
  - Introduces further orientation variability.

---

#### Photometric Transformations
- **Brightness / Contrast Adjustment**  
  - Probability: **20%**
  - Limits: ±10%
  - Improves robustness to lighting changes.

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**  
  - Probability: **20%**
  - Clip limit: 2
  - Enhances contrast in low-light or unevenly illuminated images.

- **Grayscale Conversion**  
  - Probability: **20%**
  - Simulates monochrome or low-color scenarios.

---

#### Image Degradations
Applied with **15% probability**, one of the following:
- **Gaussian Blur**
  - Kernel size: 3–5
  - Simulates motion blur or out-of-focus captures.
- **Gaussian Noise**
  - Standard deviation: 0–0.03
  - Models sensor noise in real-world photography.
- **Image Compression**
  - Quality: 75–95%
  - Simulates JPEG artifacts from compressed images.

---

### Validation and Testing Augmentations

**Pipeline:** `get_val_augmentations`

For validation and testing, **only deterministic preprocessing** is applied:
- Resizing with aspect ratio preservation
- Padding to **224 × 224**

No random augmentations are used to ensure **consistent and fair evaluation**.

---

### Rationale

Given the **limited dataset size** (~74 images per class in the original dataset, balanced between *onlyhand* and *selfie*), data augmentation plays a critical role in preventing overfitting and improving generalization.

The selected augmentations specifically address:
- **Geometric variability**  
  (hand position, rotation, scale, distance from camera)
- **Photometric variability**  
  (lighting conditions, contrast differences)
- **Image quality degradation**  
  (blur, noise, compression artifacts)

This aligns with the project’s goal of robust hand gesture recognition under diverse real-world conditions.

---

### Normalization

After augmentations, images are normalized to the **[-1, 1] range** using TensorFlow’s  
`preprocess_input` for **MobileNetV2** compatibility.

- **Location:** `utils/classification_generator.py`
- **Stage:** Applied after augmentation and before batching
- **Purpose:** Ensures correct input scaling for pretrained MobileNetV2 weights

---

This augmentation pipeline significantly improves model robustness while maintaining a clean, reproducible, and storage-efficient training workflow. 

## 5. Model Architectures

This section describes the **model architectures** used in the two main stages of the project: **object detection** (localizing the hand in an image) and **classification** (predicting the digit gesture from the cropped hand region).  
Models were selected based on **efficiency**, **task suitability**, and **project recommendations**, and were trained using a consistent and reproducible pipeline.

---

### Stage 1: Object Detection

- **Model**: **YOLOv11**  
  Fine-tuned from a pretrained checkpoint using the **Ultralytics** framework.

- **Configuration**:  
  Defined in `configuration/detection.yaml`, which specifies:
  - Paths to the **train / validation / test** splits generated by  
    `utils/split_object_detection.py`
  - **Number of classes**: 1  
  - **Class name**: `hand`

  Training experiments and evaluation are documented in:  
  `notebooks/01-yolo_model_training.ipynb`

- **Why YOLOv11**:
  - High efficiency and strong performance in **single-class object detection**
  - Robust to scale changes, background complexity, and camera distance
  - Suitable for both *onlyhand* and *selfie* images
  - Part of the YOLO family, which is explicitly allowed by the project guidelines

- **Saved Models** (`models/detection/`):
  - **`hand_detector_best.pt`**  
    Best-performing checkpoint selected based on validation metrics  
    (mAP@50–95 ≈ **0.99**)
  - **`hand_detector_last.pt`**  
    Final checkpoint at the end of training, useful for further experimentation or fine-tuning

---

## Stage 2: Classification

- **Model**: **MobileNetV2**  
  Pretrained on **ImageNet**, trained end-to-end with a custom classification head.

- **Architecture Definition**:  
  Implemented in `utils/classification_model_builder.py`.

  **Base network**:
  - MobileNetV2 (`include_top=False`)
  - Input shape: **224 × 224 × 3**
  - Base layers frozen by default, except the last `trainable_layers` if > 0

  **Classification head**:
  - `GlobalAveragePooling2D`
  - `Dense(256, activation="relu", kernel_regularizer=L2(0.01))`
  - `BatchNormalization`
  - `Dropout(rate=0.5)`
  - `Dense(6, activation="softmax", kernel_regularizer=L2(0.01), name="predictions")`

- **Training Pipeline**:
  - Data augmentation: `utils/classification_augmentations.py`
  - Data loading and normalization: `utils/classification_generator.py`
  - Dataset splits:
    - Original dataset: `utils/split_classification.py`
    - Augmented dataset (with additional images): `utils/split_classification_added_elvin.py`
  - Training and evaluation notebook:  
    `notebooks/02-classification_model_training.ipynb`

- **Training Strategy and Multiple Runs**:
  
  All classification models were trained using the **same architecture and training pipeline**.  
  Multiple training runs were performed **without changing the model structure**, allowing the inherent **stochasticity of deep learning training** (random initialization, data shuffling, augmentation randomness) to lead the optimizer toward **different local minima**.  

  This strategy resulted in progressively improved performance across runs, even with identical settings.

- **Why MobileNetV2**:
  - Lightweight and computationally efficient
  - Well-suited for small-to-medium datasets
  - Strong transfer learning capabilities due to ImageNet pretraining
  - Effective at handling variations in hand shape, lighting, and orientation

- **Saved Models** (`models/classification/`):
  - **`mobilenetv2_best.keras`**  
    Result of the **first training run** on the original dataset  
    ~91% validation accuracy, ~0.54 loss
  - **`mobilenetv2_best_new.keras`**  
    Best model obtained from **subsequent training runs** using the same pipeline  
    Improved accuracy (~93%), attributed to better convergence to a local minimum
  - **`mobilenetv2_best_added_elvin.keras`**  
    Model trained after **augmenting the dataset with additional images from Elvin Mahmudzada**  
    Achieved the best performance: ~95% validation accuracy with lower loss (~0.39)

---

Overall, this two-stage architecture combines **accurate hand localization** with **efficient and robust gesture classification**, while maintaining a clean experimental setup that highlights the impact of both **training stochasticity** and **dataset expansion** on model performance.
