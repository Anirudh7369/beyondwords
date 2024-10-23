import os
import numpy as np
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.models import load_model

# Load your trained model
model = load_model('my_model.h5')  # Ensure the model path is correct

# Define the validation data directory
validation_dir = 'C:/Users/ASUS/PycharmProjects/BeyondWords/validation'  # Adjust this path

# Set up data augmentation
datagen = ImageDataGenerator(rescale=1./255)

# Create a validation generator
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)

# Convert accuracy to percentage
accuracy_percentage = accuracy * 100

print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy_percentage:.2f}%')  # Display accuracy as percentage
