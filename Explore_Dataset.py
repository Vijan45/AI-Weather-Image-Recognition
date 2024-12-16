import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Path to dataset
dataset_path = "weather_dataset/"

# Categories
categories = os.listdir(dataset_path)
print("Categories:", categories)

# Displaying a few samples
for category in categories:
    category_path = os.path.join(dataset_path, category)
    sample_image = os.listdir(category_path)[0]
    img = cv2.imread(os.path.join(category_path, sample_image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure()
    plt.imshow(img)
    plt.title(category)
    plt.axis('off')
plt.show()