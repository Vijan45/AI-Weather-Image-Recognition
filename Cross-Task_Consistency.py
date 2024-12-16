from sklearn.metrics import accuracy_score

# Example: Simulate some metrics for tasks
# Replace these with actual metrics from your model evaluations
metrics = {
    'Multiclass': {'Accuracy': 0.85},
    'Multilabel': {'Accuracy': 0.78},
    'Image-Text': {'Accuracy': 0.82},
}

# Calculate Adaptability Index
baseline_accuracy = 0.5  # Assume baseline accuracy (e.g., random guessing)
adaptability_index = {}

# Ensure metrics dictionary exists and contains necessary accuracy values
for task, result in metrics.items():
    adaptability_index[task] = (result['Accuracy'] - baseline_accuracy) / baseline_accuracy

print("Adaptability Index:", adaptability_index)

