import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

# Define categories for classification
categories = ['Dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Set up data generators for training
datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    'C:/Users/bhija/Documents/1AAA23/weather_dataset',  # Correct path to dataset
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # Use sparse for integer labels
)

# Show the label distribution
label_counts = pd.Series(train_generator.classes).value_counts()

# Plot label distribution
label_counts.plot(kind='bar', color='skyblue')
plt.title("Label Distribution in Training Set")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.show()

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers to use the model for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(categories), activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10)

# Save the model
model.save('weather_classifier_transfer_learning.h5')

# Optionally, plot the training accuracy/loss curves
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.title('Training Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.show()
