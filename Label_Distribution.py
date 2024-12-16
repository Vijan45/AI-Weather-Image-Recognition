import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define categories
categories = ['Dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Set up your ImageDataGenerator to load the data
datagen = ImageDataGenerator(rescale=1./255)

# Assuming you have a directory with subdirectories for each category
train_generator = datagen.flow_from_directory(
    'path_to_train_directory',  # Replace with the actual path to your training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # Or 'categorical', based on your setup
)

# Get the labels (y_train)
y_train = train_generator.classes

# Count label occurrences in training set
label_counts = pd.Series(y_train).value_counts()

# Map numeric labels to category names
label_counts.index = label_counts.index.map(lambda x: categories[x])

# Plot label distribution
label_counts.plot(kind='bar', color='skyblue')
plt.title("Label Distribution in Training Set")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()
