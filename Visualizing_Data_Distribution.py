import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Define the categories (weather classes)
categories = ['Dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Load the preprocessed data and labels if already saved
data = np.load('data.npy')
labels = np.load('labels.npy')

# Assuming the train-test split has already been done
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Visualize class distribution
label_counts = pd.Series(y_train).value_counts().sort_index()
sns.barplot(x=categories, y=label_counts)
plt.xticks(rotation=45)
plt.title("Training Data Class Distribution")
plt.xlabel("Weather Categories")
plt.ylabel("Count")
plt.show()