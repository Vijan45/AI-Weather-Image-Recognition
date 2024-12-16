import tensorflow as tf
from tensorflow.keras.datasets import cifar10  # Example dataset
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the data (scale pixel values to [0, 1])
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Print dataset shape to verify
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Apply augmentation to a sample image
augmented_images = next(datagen.flow(X_train[:1], batch_size=1))

# Visualize augmented images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(augmented_images[0].astype('uint8'))
    plt.axis('off')
plt.show()
