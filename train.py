import random
import tensorflow as tf
from tf_keras.models import Model
from tf_keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Multiply
from tf_keras.applications import MobileNetV2
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tf_keras import backend as k
import matplotlib.pyplot as plt
import gc
import numpy as np

# Clear Keras session and garbage collect
k.clear_session()
gc.collect()

# Set seeds for reproducibility
random.seed(30)
tf.random.set_seed(30)

# Dataset paths
DATASET_BASE_PATH = r"C:\Users\ASUS\PycharmProjects\BeyondWords\my_dataset"

# Gesture categories
list_of_gestures = ['5', '1', '2', '7', '6', '3', '9', '4', '8', 'v', 'c']

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

# Training and validation generators
train_generator = datagen.flow_from_directory(
    DATASET_BASE_PATH,
    target_size=(224, 224),
    batch_size=16,
    class_mode='sparse',
    subset='training',
    shuffle=True,
    seed=30
)

validation_generator = datagen.flow_from_directory(
    DATASET_BASE_PATH,
    target_size=(224, 224),
    batch_size=16,
    class_mode='sparse',
    subset='validation',
    shuffle=False
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

# Focal Loss function
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',  # Replace with focal_loss(alpha, gamma) for focal loss
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_hierarchical_model.h5', save_best_only=True, monitor='val_loss')
]

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# Fine-tune the base model (unfreeze last 30 layers)
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save the final model
model.save('hand_gesture_recognition_hierarchical.h5')
print("Model saved as 'hand_gesture_recognition_hierarchical.h5'")

# Plot accuracy and loss
def plot_training_history(history, title_prefix=""):
    # Accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{title_prefix} Training and Validation Accuracy')
    plt.show()

    # Loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title_prefix} Training and Validation Loss')
    plt.show()

# Plot histories
plot_training_history(history, "Initial Training")
plot_training_history(history_finetune, "Fine-Tuning")
