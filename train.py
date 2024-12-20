import random
import tensorflow as tf
from tf_keras.models import Model
from tf_keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Multiply
from tf_keras.applications import MobileNetV2
from tf_keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tf_keras.optimizers import Adam
from tf_keras import backend as k
import gc
import numpy as np

# Clear session and garbage collect
k.clear_session()
gc.collect()

# Reproducibility
random.seed(30)
tf.random.set_seed(30)

# Dataset paths
DATASET_BASE_PATH = r"C:\Users\ASUS\PycharmProjects\BeyondWords\train_dataset"

# Gesture categories
list_of_gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'Allah', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Namaste', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    fill_mode='nearest'
)

# Training generator
train_generator = datagen.flow_from_directory(
    DATASET_BASE_PATH,
    target_size=(224, 224),
    batch_size=16,
    class_mode='sparse',
    subset='training',
    shuffle=True,
    seed=30
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Base model: MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

# Attention mechanism
def attention_layer(inputs):
    attention = Dense(inputs.shape[-1], activation='softmax')(inputs)
    return Multiply()([inputs, attention])

# Build model
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = attention_layer(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(len(list_of_gestures), activation='softmax')(x)

model = Model(inputs, outputs)

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model (weights optimization)
history = model.fit(
    train_generator,
    epochs=20,
    class_weight=class_weights
)

# Save model weights
model.save_weights('Final_trained_weights.h5')
print("Model weights saved as 'Final_trained_weights.h5'")
