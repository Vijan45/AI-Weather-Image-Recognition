# Predict test images
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Visualize predictions
def visualize_predictions(images, true_labels, predicted_labels, num_samples=10):
    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {categories[true_labels[i]]}\nPred: {categories[predicted_labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions(X_test[:10], y_test[:10], predicted_labels[:10])