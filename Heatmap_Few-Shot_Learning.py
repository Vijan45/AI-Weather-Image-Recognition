import seaborn as sns

# Plot a heatmap of distances
sns.heatmap(distances[:10], annot=True, cmap='coolwarm', xticklabels=categories, yticklabels=False)
plt.title("Few-Shot Learning: Distance to Class Prototypes")
plt.xlabel("Classes")
plt.ylabel("Samples")
plt.show()

