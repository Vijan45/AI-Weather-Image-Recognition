import matplotlib.pyplot as plt
import pickle  # To load your preprocessed dataset
from Preprocessing_Images import preprocess_images  # Assuming the preprocess function is imported
import numpy as np

# Define categories (weather classes)
categories = ['Dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Your dataset path
dataset_path = r'C:\Users\bhija\Documents\1AAA23\weather_dataset'

# Preprocess the images and get data
data, labels = preprocess_images(dataset_path)

# Split the dataset into training, validation, and test sets
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create a dictionary for class descriptions
class_descriptions = {
    "dew": "A layer of water droplets that forms on cool surfaces overnight.",
    "fogsmog": "A thick, cloud-like mass near the ground, reducing visibility.",
    "frost": "A thin, icy coating that forms on surfaces during cold conditions.",
    "glaze": "A smooth layer of ice covering surfaces due to freezing rain.",
    "hail": "Small, round ice pellets that fall during intense storms.",
    "lightning": "A bright flash of light caused by an electrical discharge during storms.",
    "rain": "Water droplets falling from clouds to the ground.",
    "rainbow": "A colorful arc of light formed after rain, caused by refraction.",
    "rime": "A frost-like deposit of ice crystals formed in freezing fog.",
    "sandstorm": "A cloud of sand particles carried by strong winds in arid regions.",
    "snow": "Soft, white flakes of frozen water vapor falling from the sky."
}

# Example: Pair image with description
example_image = X_train[0]
example_label = y_train[0]
example_description = class_descriptions[categories[example_label]]

print(f"Label: {categories[example_label]}")
print(f"Description: {example_description}")
plt.imshow(example_image)
plt.axis('off')
plt.show()