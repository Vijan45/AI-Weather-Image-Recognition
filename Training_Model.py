import tensorflow as tf
from sklearn.model_selection import train_test_split  # Import train_test_split
from Building_CNN_Model import create_cnn_model  # Import the model creation function
from Preprocessing_Images import preprocess_images  # Import the preprocessing function
import pickle  # Import pickle to save history

# Define categories (weather classes)
categories = ['Dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
dataset_path = r'C:\Users\bhija\Documents\1AAA23\weather_dataset'  # Your dataset path

# Preprocess the images and split the dataset into training, validation, and test sets
data, labels = preprocess_images(dataset_path)
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the model
input_shape = (128, 128, 3)  # Image dimensions
num_classes = len(categories)  # Number of classes
model = create_cnn_model(input_shape, num_classes)  # Create the CNN model

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,  # You can adjust the number of epochs
    batch_size=32,
    verbose=1
)

# Save history to a file
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("Training complete and history saved.")