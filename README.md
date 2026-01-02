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

---

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

---

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

---

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

---

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

---

## 6. Training Process

This section describes the training pipelines for both **object detection** and **image classification**, focusing on data preparation, model configuration, and the rationale behind the chosen training parameters. The implementation is provided in the Jupyter notebooks located in the `notebooks/` directory. Quantitative and qualitative results are intentionally omitted here and are discussed in a dedicated section.

---

### Detection Training  
**Notebook**: `notebooks/01-yolo_model_training.ipynb`

#### Workflow
The object detection dataset is extracted from `data/data.zip` and split into **training, validation, and test** subsets using `utils/split_object_detection.py`. This utility preserves the YOLO-compatible directory layout (`images/` and `labels/`) for each subset, ensuring direct compatibility with the Ultralytics training pipeline.

The detection model is trained using **YOLOv11n**, selected for its favorable trade-off between accuracy and computational efficiency. All model- and augmentation-related settings are defined in `configuration/detection.yaml`, allowing the training process to remain reproducible and easily configurable. After training, the model is evaluated on the validation and test sets to assess generalization performance (evaluation details are reported in a separate section).

#### Training Configuration and Parameter Rationale
- **Pretrained weights (`yolo11n.pt`)**  
  Initializing from pretrained weights enables faster convergence and better generalization by transferring low-level and mid-level visual features learned from large-scale datasets.

- **Epochs (100)**  
  A sufficiently large number of epochs is used to allow the model to converge, while early stopping prevents unnecessary overfitting.

- **Batch size (16)**  
  Chosen as a balance between GPU memory constraints and gradient stability. This batch size allows effective training on high-resolution images (640×640).

- **Optimizer (AdamW)**  
  AdamW is used instead of standard Adam to decouple weight decay from gradient updates, improving generalization and training stability.

- **Initial learning rate (`lr0 = 0.001`)**  
  A moderate starting learning rate that works well with AdamW, enabling steady convergence without unstable updates.

- **Cosine learning rate scheduler (`cos_lr = True`)**  
  Gradually reduces the learning rate following a cosine curve, encouraging smoother convergence and helping the model settle into a better local minimum.

- **Image size (640 × 640)**  
  Provides a good compromise between detection accuracy for small objects (hands) and computational cost.

- **Device (GPU, `device = 0`)**  
  Training is performed on a single GPU to significantly accelerate both forward and backward passes.

- **Early stopping patience (15)**  
  Stops training if validation performance does not improve for 15 consecutive epochs, reducing overfitting and unnecessary computation.

- **Automatic Mixed Precision (`amp = True`)**  
  Uses FP16 where possible to speed up training and reduce GPU memory usage without sacrificing accuracy.

---

### Classification Training  
**Notebook**: `notebooks/02-classification_model_training.ipynb`

#### Workflow
Classification experiments are conducted on two dataset variants:
- `data/data.zip` — the original dataset
- `data/data_added_elvin.zip` — the original dataset augmented with additional samples from *Elvin Mahmudzada’s* dataset

Data is split into **train, validation, and test** sets using:
- `utils/split_classification.py` for the original dataset
- `utils/split_classification_added_elvin.py` for the augmented dataset

All splits follow a **0.7 / 0.2 / 0.1** ratio and explicitly enforce class balance across all six classes (0–5). Additional constraints ensure balanced representation of image types (`onlyhand`, `selfie`) and, for the augmented dataset, a balanced contribution of the externally added samples.

Data loading and preprocessing are handled by `utils/classification_generator.py`. Images are normalized to the **[-1, 1]** range, as required by MobileNetV2. Training-time augmentations are applied via `utils/classification_augmentations.py`, while validation and test data undergo only deterministic resizing to ensure unbiased evaluation.

The classification model is defined in `utils/classification_model_builder.py` and is based on **MobileNetV2** with custom fully connected head layers.

#### Training Configuration and Parameter Rationale
- **Input resolution (224 × 224 × 3)**  
  Matches the default input size for MobileNetV2, ensuring compatibility with pretrained weights.

- **Number of classes (6)**  
  Corresponds directly to the six gesture categories in the dataset.

- **Two-stage training strategy**  
  - **Stage 1: Frozen backbone**  
    The MobileNetV2 base is frozen to train only the newly added classification head, allowing the model to adapt to the task without disturbing pretrained features.
  - **Stage 2: Fine-tuning**  
    The last `NUM_FT_LAYERS` layers of the backbone are unfrozen to refine higher-level features for the specific classification task. Batch Normalization layers remain frozen to preserve stable statistics.

- **Epochs (50 per stage)**  
  Provides sufficient time for convergence in both the head training and fine-tuning phases.

- **Batch size (16)**  
  Chosen to balance GPU memory usage and gradient stability.

- **Optimizer (Adam)**  
  Adam is well-suited for transfer learning scenarios due to its adaptive learning rate behavior.

- **Base learning rate (`BASE_LR = 0.008115748913403031`)**  
  A relatively higher learning rate is used during head training to allow rapid adaptation of newly initialized layers.

- **Fine-tuning learning rate (`FT_LR = 9.51e-05`)**  
  A much smaller learning rate is used during fine-tuning to avoid destroying pretrained representations while still allowing task-specific adaptation.

- **Dropout rate (`0.583`)**  
  A relatively high dropout value is applied to reduce overfitting, which is especially important given the moderate dataset size.

- **Number of fine-tuned layers (`NUM_FT_LAYERS = 39`)**  
  Selected to fine-tune only the higher-level semantic features while keeping low-level features (edges, textures) fixed.

- **Callbacks**  
  - **EarlyStopping**: Stops training when validation loss no longer improves, helping prevent overfitting.  
  - **ModelCheckpoint**: Saves the best-performing model according to validation loss.  
  - **ReduceLROnPlateau**: Dynamically reduces the learning rate when progress stalls, enabling finer convergence.

---

## 7. Evaluation Results

This section presents the quantitative evaluation of the trained models for **object detection** and **image classification**. All metrics are reported on validation and test splits and are intended to demonstrate model accuracy, robustness, and generalization capability. Detailed training logs, curves, and visualizations are available in the corresponding Jupyter notebooks.

---

### Object Detection (YOLOv11)  
**Notebook**: `notebooks/01-yolo_model_training.ipynb`

The object detection model is trained to localize hands in both *only-hand* and *selfie* images. Performance is evaluated using standard COCO-style metrics.

#### Evaluation Metrics
- **mAP@0.5**: **0.995** (validation set)
- **Precision**: **~0.9994**
- **Recall**: **1.00**
- **mAP@0.5:0.95**: **~0.990–0.991**

#### Saved Results
- **Training metrics and curves** (loss, precision, recall, mAP per epoch):  
  `results/detection_metrics/training_hand_yolo11/`
- **Independent test evaluation metrics**:  
  `results/detection_metrics/val_test_metrics/`

#### Key Findings
The YOLOv11 model demonstrates consistently high performance across all evaluation metrics. Near-perfect recall indicates that the model successfully detects all true hand instances, while very high precision confirms a minimal number of false positives. Strong mAP values across multiple IoU thresholds show that the predicted bounding boxes are both accurate and well-aligned with ground truth annotations. The model generalizes well to different hand poses, image compositions, and background complexity, making it suitable for real-time hand localization tasks.

The trained detection models are stored in:
- `models/detection/hand_detector_best.pt` — best-performing model
- `models/detection/hand_detector_last.pt` — model from the final training epoch

---

### Classification (MobileNetV2)  
**Notebook**: `notebooks/02-classification_model_training.ipynb`

The classification model predicts one of six hand gesture classes based on cropped hand images obtained from the detection stage. Evaluation is performed on a balanced test set.

#### Overall Performance
- **Test Accuracy**: **94.9% (0.949)**

#### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|-------:|---------:|--------:|
| 0 | 1.000 | 1.000 | 1.000 | 13 |
| 1 | 0.929 | 1.000 | 0.963 | 13 |
| 2 | 1.000 | 0.846 | 0.917 | 13 |
| 3 | 0.857 | 0.923 | 0.889 | 13 |
| 4 | 0.923 | 0.923 | 0.923 | 13 |
| 5 | 1.000 | 1.000 | 1.000 | 13 |
| **Macro Avg** | **0.951** | **0.949** | **0.949** | **78** |
| **Weighted Avg** | **0.951** | **0.949** | **0.949** | **78** |

#### Confusion Matrix Analysis
The confusion matrix (visualized in the notebook) shows that most predictions lie on the main diagonal, indicating strong class separability. Minor misclassifications occur primarily between visually similar gestures, particularly **class 2** and **class 3**, which differ only by a single extended finger. These errors are likely caused by subtle pose variations, partial occlusions, or viewpoint differences.

#### Key Findings
Initial training runs achieved approximately **91% accuracy**, after which further improvements became limited. This behavior indicates that the model was approaching its performance ceiling given the relatively small size of the available dataset. With only **444 images** in the original dataset, the model quickly learned the dominant visual patterns but lacked sufficient variability to continue improving its generalization capability.

Subsequent experiments increased accuracy to around **93%**, but this improvement came with a higher loss value, suggesting partial overfitting rather than a true gain in generalization. This further supports the hypothesis that dataset size, rather than model architecture, was the primary limiting factor at this stage.

Incorporating additional samples from *Elvin Mahmudzada’s* dataset—**108 images in total (18 per class)**—resulted in a clear and measurable improvement. Despite the relatively small increase in data volume, test accuracy rose to nearly **95%**, while the loss value decreased significantly. This confirms that even a modest expansion of the dataset can push the model beyond its previous performance ceiling. Given this trend, it is reasonable to expect that with further dataset expansion, the same MobileNetV2-based architecture could achieve **accuracy levels approaching 99%** without requiring architectural changes.

Beyond overall accuracy, a qualitative analysis revealed an important dataset bias. In the original dataset, images of **female hands** were underrepresented, making gesture recognition on female hands more challenging for the model. Interestingly, the model trained **without** Elvin’s data (93% accuracy) performed noticeably better on female hands, while the model trained **with** Elvin’s data (95% accuracy) showed improved performance on male hands but degraded performance on female hands. This behavior can be explained by the fact that Elvin’s dataset predominantly contains images of **male hands**, further amplifying the existing imbalance.

These observations highlight that the higher overall accuracy of the final model does not necessarily imply uniformly better performance across all demographic subsets. A more balanced dataset—particularly with a larger number of female hand samples—would likely improve robustness and fairness across different hand types. Unfortunately, due to time constraints, limited access to suitable participants, and the lack of publicly available datasets closely matching the characteristics of the existing data, such an expansion was not feasible. Importantly, no externally sourced data was added that could introduce uncontrolled bias, ensuring that all reported results remain methodologically consistent and comparable.

The trained classification models are stored in:
- `models/classification/mobilenetv2_best.keras` — initial training run
- `models/classification/mobilenetv2_best_new.keras` — subsequent runs
- `models/classification/mobilenetv2_best_added_elvin.keras` — final model with added data  

---

## 8. Inference Pipeline

This section describes the **end-to-end inference pipeline** for hand gesture recognition. The pipeline combines the trained **YOLOv11** object detection model and the **MobileNetV2** classification model to first localize hands in an image and then predict the corresponding gesture class (**0–5**, representing finger counts as well as poses such as *fist* and *open palm*).

---

### Implementation Overview

The inference workflow is demonstrated in  
`notebooks/03-inference_pipeline.ipynb`,  
with the core, reusable logic implemented in  
`utils/inference.py`.

The pipeline supports both **single-image** and **batch-image** inference and visualizes results using OpenCV (`cv2_imshow`), making it suitable for interactive experimentation in Google Colab.

---

### Inference Workflow

1. **Model Loading**  
   The pipeline loads the best-performing trained models:
   - **Detection**: `models/detection/hand_detector_best.pt` (YOLOv11)  
   - **Classification**: `models/classification/mobilenetv2_best_added_elvin.keras` (MobileNetV2)

2. **Image Input and Hand Detection**  
   Input image(s) are read using OpenCV.  
   YOLOv11 inference is executed with a **confidence threshold of 0.15** to detect hand bounding boxes.  
   If no hands are detected in an image, the pipeline skips further processing for that image and logs a warning.

3. **Bounding Box Selection and Preprocessing**  
   From all detected hands, the bounding box with the **highest confidence score** is selected.  
   The corresponding region is cropped and processed as follows:
   - Resize to **224 × 224**
   - Convert from BGR to RGB
   - Normalize pixel values to the **[-1, 1]** range (MobileNetV2 requirement)

4. **Gesture Classification**  
   The preprocessed hand crop is passed to the classification model, which outputs a predicted gesture class in the range **0–5**.

5. **Visualization and Output**  
   - The predicted class label (e.g., `Predicted: 3`) is drawn on the original image together with the detected bounding box.
   - Text placement is adjusted dynamically to ensure readability (above or below the box).
   - Both the annotated original image and the cropped hand region are displayed using `cv2_imshow`.
   - The pipeline returns a dictionary mapping image paths to predicted classes:
     ```
     {image_path: predicted_class}
     ```

---

### Testing and Usage

The pipeline is designed to work on **previously unseen images**.  
In the notebook, it is demonstrated on sample inputs (e.g., `/content/test_image_1.jpg` to `/content/test_image_5.jpg`). Users or instructors can easily test their own images by replacing the `image_paths` list with custom file paths and re-running the inference cell.

The implementation includes basic error handling for:
- invalid or missing image paths,
- images where no hands are detected.

Batch processing is supported, enabling efficient inference over multiple images in a single run.

---

### Model Selection and Flexibility

By default, the pipeline uses the models that achieved the best empirical performance:
- `hand_detector_best.pt` for detection
- `mobilenetv2_best_added_elvin.keras` for classification (≈95% test accuracy)

However, the design is modular, allowing any of the other saved models in `models/detection/` or `models/classification/` to be swapped in with minimal code changes. This makes the pipeline easy to extend, compare models, or adapt for deployment scenarios.

---

Overall, the inference pipeline provides a clean and modular integration of detection and classification, closely resembling a real-time gesture recognition system. Its structure allows straightforward reuse in downstream applications such as interactive systems, human–computer interaction tasks, or accessibility-oriented solutions.

---

## 9. Conclusion

This project successfully implemented a two-stage pipeline for hand gesture recognition, achieving robust performance in detecting and classifying hand poses representing digits 0–5. Key accomplishments include:
- **High Accuracy**: The object detection model reached an mAP@50-95 of 0.994, while the classification model attained 95% accuracy on the test set after dataset augmentation.
- **Real-World Robustness**: Through careful data collection, augmentation strategies, and model fine-tuning, the system handles variations in lighting, backgrounds, hand orientations, and skin tones effectively.
- **Modular Design**: The reproducible structure, with separate utilities, notebooks, and configurations, facilitates easy extension and experimentation.
- **Lessons Learned**: Dataset size and diversity were critical limiting factors; biases (e.g., underrepresentation of female hands) highlighted the importance of balanced data. Stochastic training variations and modest data sharing significantly improved results without architectural changes.

Future work could involve expanding the dataset for better demographic balance, integrating real-time video processing, or exploring advanced models like Vision Transformers for further accuracy gains. Overall, this project demonstrates practical computer vision skills and provides a solid foundation for gesture-based applications in HCI, accessibility, and beyond.

---

## 10. License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code and models for any purpose, provided that the original attribution is preserved.

### MIT License

Copyright (c) 2026 Kamal Mustafayev

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**Note on Data**: The datasets (`data.zip` and `data_added_elvin.zip`) are provided strictly for educational purposes only and may not be used, copied, modified, distributed, or redistributed in any form without explicit written permission from the author (Kamal Mustafayev). Shared data from Elvin Mahmudzada is used with permission and is subject to the same restrictions. Unauthorized use of the data is prohibited. If you wish to use the data, please contact the author for approval.
