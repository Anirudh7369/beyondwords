import random
import tensorflow as tf
from tf_keras.models import Model
from tf_keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Multiply
from tf_keras.applications import MobileNetV2
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.optimizers import Adam
import matplotlib.pyplot as plt
import gc
import numpy as np
from tf_keras import backend as k
from tf_keras.models import load_model


# Clear Keras session and garbage collect
k.clear_session()
gc.collect()

# Set seeds for reproducibility
random.seed(30)
tf.random.set_seed(30)

# Validation dataset path (provide this path)
VALIDATION_DATASET_PATH = r"C:\Users\ASUS\PycharmProjects\BeyondWords\Validation dataset"  # Input the validation dataset path here

# Gesture categories
list_of_gestures = ['5', '1', '2', '4', '3', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

# Data augmentation (for validation, we only rescale images)
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Validation generator
validation_generator = datagen.flow_from_directory(
    VALIDATION_DATASET_PATH,  # Use the provided validation dataset path here
    target_size=(224, 224),
    batch_size=16,
    class_mode='sparse',
    shuffle=False
)

# Base model: MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model for initial training

# Attention mechanism
def attention_layer(inputs):
    attention = Dense(inputs.shape[-1], activation='softmax')(inputs)
    return Multiply()([inputs, attention])

# Build the model with feature hierarchy and attention
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = attention_layer(x)  # Add attention layer
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(len(list_of_gestures), activation='softmax')(x)  # Output layer

model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load the best saved model
model = load_model("hand_gesture_recognition_hierarchical.h5")

# Evaluate the model on the validation set
validation_loss, validation_accuracy = model.evaluate(validation_generator, verbose=1)

# Print validation accuracy and loss
print(f"Validation Accuracy: {validation_accuracy:.4f}")
print(f"Validation Loss: {validation_loss:.4f}")
