from sklearn.metrics import pairwise_distances
import numpy as np

# Extract feature embeddings using the pre-trained ResNet50 base model
embedding_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Generate embeddings for a few labeled examples
few_shot_X_train, few_shot_y_train = X_train[:50], y_train[:50]  # Use 50 samples for few-shot learning
train_embeddings = embedding_model.predict(few_shot_X_train)

# Generate embeddings for the validation set
val_embeddings = embedding_model.predict(X_val)

# Calculate class prototypes (mean embeddings for each class)
prototypes = []
for category in categories:
    category_indices = [i for i, y in enumerate(few_shot_y_train) if y == category]
    category_embeddings = train_embeddings[category_indices]
    prototypes.append(np.mean(category_embeddings, axis=0))

prototypes = np.array(prototypes)

# Classify validation samples based on distances to prototypes
distances = pairwise_distances(val_embeddings, prototypes)
predictions = np.argmin(distances, axis=1)

# Evaluate the few-shot model
accuracy = np.mean(predictions == np.argmax(y_val, axis=1))
print(f"Few-Shot Learning Accuracy: {accuracy:.2f}")