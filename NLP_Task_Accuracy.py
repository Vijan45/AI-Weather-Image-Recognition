# Simulate model accuracy with clean vs. noisy sentiment labels
clean_sentiment_accuracy = 0.92
noisy_sentiment_accuracy = 0.75

# Bar chart comparison
import matplotlib.pyplot as plt

labels = ['Clean Labels', 'Noisy Labels']
accuracies = [clean_sentiment_accuracy, noisy_sentiment_accuracy]

plt.bar(labels, accuracies, color=['green', 'red'])
plt.title('Impact of High-Quality Labels in Sentiment Analysis')
plt.ylabel('Accuracy')
plt.show()