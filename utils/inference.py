import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from utils.classification_augmentations import get_val_augmentations
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def infer_hand_gestures(image_paths, det_model, cls_model):
    """
    Predict hand gestures for multiple images and display original image
    with bounding box and predicted digit, ensuring the text is visible.
    
    Args:
    - image_paths: list of str, paths to the images
    - det_model: YOLO detection model
    - cls_model: Keras classification model
    
    Returns:
    - dict: {image_path: predicted_class} for each image
    """
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    
    predictions = {}
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image from {image_path}. Skipping.")
            continue
        
        original_image = image.copy()
        results = det_model(image, conf=0.15)
        
        if len(results[0].boxes) == 0:
            print(f"Warning: No hand detected in {image_path}. Skipping.")
            continue
        
        best_box = results[0].boxes[0]
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
        
        # Crop for classification (можно оставить без изменения размера)
        crop = image[y1:y2, x1:x2]
        
        # Validation augmentations
        val_augs = get_val_augmentations(img_size=224)
        augmented = val_augs(image=crop)
        crop_aug = augmented['image']
        
        # Convert to RGB and preprocess for MobileNetV2
        crop_rgb = cv2.cvtColor(crop_aug, cv2.COLOR_BGR2RGB)
        crop_processed = preprocess_input(crop_rgb.astype(np.float32))
        
        # Predict gesture
        pred = cls_model.predict(np.expand_dims(crop_processed, axis=0))
        predicted_class = np.argmax(pred)
        
        # Draw bounding box
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare text and ensure it fits in the image
        text = f"Predicted: {predicted_class}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75  
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1
        text_y = y1 - 5  
        if text_y - text_size[1] < 0:
          text_y = y1 + text_size[1] + 5

        cv2.putText(original_image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
        # Display results
        print(f"Original Image with Bounding Box and Prediction for {image_path}:")
        cv2_imshow(original_image)
        
        print(f"Cropped Hand Region for {image_path}:")
        cv2_imshow(crop)
        
        predictions[image_path] = predicted_class
    
    return predictions
