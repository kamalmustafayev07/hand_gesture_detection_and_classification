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

## 2. Dataset Description
