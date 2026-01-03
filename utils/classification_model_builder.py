import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def build_transfer_mobilenetv2(
    input_shape=(224, 224, 3),
    num_classes=6,
    dropout_rate=0.5,
    trainable_layers=20  # Number of layers to fine-tune (0 for frozen base)
):
    """
    Builds a transfer learning model using MobileNetV2 as base.
    
    Args:
        input_shape (tuple): Input image shape (H, W, C)
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        trainable_layers (int): Number of last layers to make trainable for fine-tuning.
                                Set to 0 to freeze the entire base model.
    
    Returns:
        model (tf.keras.Model): Transfer learning model
    """
    
    # Load pre-trained MobileNetV2 base (without top classifier)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # If fine-tuning, unfreeze the last N layers
    if trainable_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    
    # Build the model
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)  # Ensure BN behaves correctly if frozen
    
    # Add classifier head with L2 regularization
    x = layers.GlobalAveragePooling2D()(x)  # Reduces parameters compared to Flatten
    x = layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=l2(0.01)  # L2 
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(
        num_classes, 
        activation='softmax', 
        name="predictions",
        kernel_regularizer=l2(0.01) # L2
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Classifier")
    
    return model