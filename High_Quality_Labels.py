import random
from sklearn.model_selection import train_test_split
from Preprocessing_Images import preprocess_images
from Building_CNN_Model import train_model, evaluate_model

# Define categories
categories = ['Dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Dataset path
dataset_path = r'C:\Users\bhija\Documents\1AAA23\weather_dataset'

# Preprocess images and split the dataset
data, labels = preprocess_images(dataset_path)
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Simulate noisy labels
noisy_labels = y_train.copy()
for i in range(int(0.2 * len(noisy_labels))):
    noisy_labels[random.randint(0, len(noisy_labels) - 1)] = random.randint(0, len(categories) - 1)

# Train models
clean_model = train_model(X_train, y_train)
noisy_model = train_model(X_train, noisy_labels)

# Evaluate models
clean_accuracy = evaluate_model(clean_model, X_test, y_test)
noisy_accuracy = evaluate_model(noisy_model, X_test, y_test)

print(f"Accuracy with Clean Labels: {clean_accuracy:.2f}")
print(f"Accuracy with Noisy Labels: {noisy_accuracy:.2f}")
