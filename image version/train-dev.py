import random
import tensorflow as tf
from tf_keras.models import Model
from tf_keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Multiply
from tf_keras.applications import MobileNetV2
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras import backend as k
import gc
import numpy as np

# Clear Keras session and garbage collect
k.clear_session()
gc.collect()
# Dataset paths and setup (reuse from Train Code)
DATASET_BASE_PATH = r"C:\Users\ASUS\PycharmProjects\BeyondWords\train_dataset"
list_of_gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'Allah', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Namaste', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

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

# Class weights (reuse from Train Code)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Rebuild base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Initially freeze the base model

# Attention mechanism
def attention_layer(inputs):
    attention = Dense(inputs.shape[-1], activation='softmax')(inputs)
    return Multiply()([inputs, attention])

# Rebuild the full model architecture
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

# Load weights from Train Code
model.load_weights('Final_trained_weights.h5')
print("Loaded weights from 'Final_trained_weights.h5'")

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Callbacks for hyperparameter tuning
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_hyperparameter_model.h5', save_best_only=True, monitor='val_loss')
]

# Hyperparameter tuning: Experiment with different learning rates
learning_rates = [1e-3, 1e-4, 1e-5]
best_accuracy = 0
best_lr = None

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model with validation data
    history_finetune = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Evaluate validation accuracy
    val_accuracy = max(history_finetune.history['val_accuracy'])
    print(f"Validation accuracy for lr={lr}: {val_accuracy}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_lr = lr

print(f"Best learning rate: {best_lr} with accuracy: {best_accuracy}")

model.save('Final_fine_tuned_model.h5')
print("Fine-tuned model saved as 'Final_fine_tuned_model.h5'")
