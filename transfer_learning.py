from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the dataset to range [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Define a function to preprocess and resize images
def preprocess_and_resize(image, label):
    """
    Resize images to 224x224 and return the image along with its label.
    """
    image = tf.image.resize(image, (224, 224))  # Resize to target size
    return image, label

# Convert CIFAR-10 data to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Apply preprocessing, shuffling, and batching to the training dataset
train_dataset = (
    train_dataset
    .map(preprocess_and_resize, num_parallel_calls=tf.data.AUTOTUNE)  # Resize and preprocess
    .shuffle(buffer_size=50000)  # Shuffle the dataset
    .batch(32)  # Batch size
    .prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch for performance
)

# Apply preprocessing and batching to the test dataset
test_dataset = (
    test_dataset
    .map(preprocess_and_resize, num_parallel_calls=tf.data.AUTOTUNE)  # Resize and preprocess
    .batch(32)  # Batch size
    .prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch for performance
)

# Load the ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers to prevent them from being updated during training
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of ResNet50
x = Flatten()(base_model.output)  # Flatten the feature maps
x = Dense(128, activation='relu')(x)  # Fully connected layer with ReLU activation
x = Dropout(0.5)(x)  # Dropout layer for regularization
output = Dense(10, activation='softmax')(x)  # Output layer for 10 classes

# Create the final model by combining base model and custom layers
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with Adam optimizer and sparse categorical cross-entropy loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for 2 epochs
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=2  # Training limited to 2 epochs
)
