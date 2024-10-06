import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

# Define CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    
    # Convolution + MaxPooling layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Prepare the data (Assuming data is organized in train/test folders)
def load_data(train_dir, test_dir, target_size=(64, 64), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_data = datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size, class_mode='binary')
    test_data = datagen.flow_from_directory(test_dir, target_size=target_size, batch_size=batch_size, class_mode='binary')
    
    return train_data, test_data

# Train the CNN
def train_cnn(train_data, test_data, input_shape, epochs=10):
    model = create_cnn_model(input_shape)
    model.fit(train_data, validation_data=test_data, epochs=epochs)
    
    return model

# Detect face in an input image
def detect_face(model, face_cascade, image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face detector and detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        
        prediction = model.predict(face_resized)
        
        # If prediction is close to 1, it means the face matches
        if prediction >= 0.5:
            label = "Matched Face"
            color = (0, 255, 0)
        else:
            label = "Unmatched Face"
            color = (0, 0, 255)
        
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the result
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to run the model
if __name__ == '__main__':
    # Set paths to dataset directories
    train_dir = 'C:/Users/kanis/OneDrive/Desktop/codes/face_matching/train'
    test_dir = 'C:/Users/kanis/OneDrive/Desktop/codes/face_matching/test'
    
    # Load the dataset
    train_data, test_data = load_data(train_dir, test_dir, target_size=(64, 64))
    
    # Train the model
    input_shape = (64, 64, 3)  # RGB images with 64x64 size
    model = train_cnn(train_data, test_data, input_shape, epochs=10)
    
    loss, accuracy = model.evaluate(test_data)

# Print the accuracy
print(f"Test Accuracy: {accuracy * 100:.2f}%")