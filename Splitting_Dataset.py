from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np

# Define categories and dataset path
categories = ['Dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
dataset_path = r'C:\Users\bhija\Documents\1AAA23\weather_dataset'  # Use raw string or escape backslashes

# Function to preprocess images
def preprocess_images(dataset_path, img_size=(128, 128)):
    data = []
    labels = []
    
    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            img = cv2.imread(file_path)
            if img is not None:  # Check if the image is loaded correctly
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize
                data.append(img)
                labels.append(label)
    
    return np.array(data), np.array(labels)

# Preprocess images to get data and labels
data, labels = preprocess_images(dataset_path)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print the shapes of the splits
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")
