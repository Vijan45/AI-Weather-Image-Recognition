from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.utils.multiclass import unique_labels

# Step 0: Load Dataset
# Using CIFAR-10 dataset as an example; replace it with your dataset if needed
(X_data, y_data), (X_test, y_test) = cifar10.load_data()

# Normalize data (scale pixel values to [0, 1])
X_data = X_data.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten y_data if needed (for multiclass models)
y_data = y_data.flatten()
y_test = y_test.flatten()

# Print dataset shape
print(f"X_data shape: {X_data.shape}, y_data shape: {y_data.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Step 1: Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Define your custom categories based on the provided chat history
custom_categories = ['Dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'snow']

# Ensure that the number of categories matches the number of labels in your data
if len(custom_categories) != 10:
    raise ValueError(f"Custom categories count should be 10, but got {len(custom_categories)}")

# Step 2: Define placeholder training functions for different models
# Replace these with your actual model training implementations
def train_multiclass_model(X_train, y_train):
    print("Training multiclass model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=2, batch_size=64)
    return model

# Step 3: Train models with different labeling schemes
models = {
    'Multiclass': train_multiclass_model(X_train, y_train),
}

# Step 4: Evaluate models
metrics = {}
for task, model in models.items():
    # Get model predictions
    predictions = model.predict(X_val).argmax(axis=1)  # Convert probability to class label
    
    # Ensure that the number of unique labels in the validation set is consistent
    unique_classes = unique_labels(y_val)
    
    # Calculate accuracy and classification report
    metrics[task] = {
        'Accuracy': accuracy_score(y_val, predictions),
        'Classification Report': classification_report(
            y_val,
            predictions,
            labels=unique_classes,  # Ensure only present labels are evaluated
            target_names=[custom_categories[i] for i in unique_classes]  # Match target names to the present classes
        )
    }

# Display accuracy and classification report for all tasks
for task, result in metrics.items():
    print(f"Task: {task}, Accuracy: {result['Accuracy']:.2f}")
    print(f"Classification Report for {task}:\n", result['Classification Report'])

# Step 5: Plot Confusion Matrix
conf_matrix = confusion_matrix(y_val, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=custom_categories, yticklabels=custom_categories)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()