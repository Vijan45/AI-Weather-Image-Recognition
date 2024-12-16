import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the categories (weather classes)
categories = ['Dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Define CNN Model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Convolutional Layers
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Flattening and Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')  # Softmax for multiclass classification
    ])
    return model

# Train the model
def train_model(X_train, y_train, input_shape=(128, 128, 3), num_classes=len(categories), epochs=20, batch_size=32):
    model = create_cnn_model(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

# Optional: Summary of the model
if __name__ == "__main__":
    input_shape = (128, 128, 3)  # Image dimensions from Part 1
    num_classes = len(categories)  # Number of weather categories
    model = create_cnn_model(input_shape, num_classes)
    model.summary()