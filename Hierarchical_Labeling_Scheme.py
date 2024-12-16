import numpy as np
from sklearn.model_selection import train_test_split
from Preprocessing_Images import preprocess_images  # Import your preprocessing function

# Define hierarchical structure
hierarchy = {
    "precipitation": ["rain", "snow", "hail"],
    "visibility": ["fogsmog", "sandstorm"],
    "ice": ["frost", "rime", "glaze"],
    "optical": ["lightning", "rainbow"],
    "dew": ["dew"]
}

# Define categories
categories = ['Dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Preprocess images and split data
dataset_path = r'C:\Users\bhija\Documents\1AAA23\weather_dataset'
data, labels = preprocess_images(dataset_path)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Assign hierarchical labels
hierarchical_labels = []
for label in y_train:
    subcategory = categories[label]
    for parent, children in hierarchy.items():
        if subcategory in children:
            hierarchical_labels.append((parent, subcategory))
            break

# Example hierarchical label
print(f"Image 1: {hierarchical_labels[0]}")
