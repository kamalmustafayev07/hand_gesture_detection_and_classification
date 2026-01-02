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

