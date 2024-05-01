# waste-classification-using-cnn
CLASSIFICATION 
 
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time

# Load the trained model
model = load_model('garbage_classification_model.h5')

# Define the class labels
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', ]

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for default camera, change if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Add a frame to the frame
    height, width, _ = frame.shape
    frame = cv2.rectangle(frame, (0, 0), (width, height), (255, 255, 255), 10)

    # Preprocess the frame for prediction
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(resized_frame, axis=0) / 255.0

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display the prediction on the frame
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Garbage Classification', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

MACHINE LEARNING MODEL
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Path to your dataset folder
dataset_path = 'Garbage'

# Split the dataset into training and testing sets
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          epochs=10,
          validation_data=validation_generator)

# Save the model
model.save('garbage_classification_model.h5')

