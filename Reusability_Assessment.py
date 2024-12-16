import matplotlib.pyplot as plt

# Example metrics (replace with actual model metrics in your use case)
metrics = {
    'Multiclass': {
        'Accuracy': 0.85,
        'Classification Report': {'precision': 0.8, 'recall': 0.7, 'f1-score': 0.75}
    },
    'Multilabel': {
        'Accuracy': 0.78,
        'Classification Report': {'precision': 0.75, 'recall': 0.72, 'f1-score': 0.73}
    },
    'Image-Text': {
        'Accuracy': 0.82,
        'Classification Report': {'precision': 0.78, 'recall': 0.75, 'f1-score': 0.76}
    }
}

# Step 1: Calculate Adaptability Index
baseline_accuracy = 0.5  # Assuming baseline accuracy of 50%

adaptability_index = {}
for task, result in metrics.items():
    adaptability_index[task] = (result['Accuracy'] - baseline_accuracy) / baseline_accuracy

# Print Adaptability Index
print("Adaptability Index:", adaptability_index)

# Step 2: Visualize Adaptability Index for each task
tasks = list(adaptability_index.keys())
adaptability_scores = list(adaptability_index.values())

plt.bar(tasks, adaptability_scores, color=['blue', 'orange', 'green'])
plt.title("Adaptability Index for Different Tasks")
plt.ylabel("Adaptability Index")
plt.xlabel("Task")
plt.ylim(0, max(adaptability_scores) + 0.1)  # Adjust y-axis limits for better visualization
plt.show()

# Step 3: Bar chart for task-specific accuracy
task_accuracies = [result['Accuracy'] for result in metrics.values()]
tasks = list(metrics.keys())

plt.bar(tasks, task_accuracies, color=['blue', 'orange', 'green'])
plt.title("Task-Specific Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Task")
plt.show()

