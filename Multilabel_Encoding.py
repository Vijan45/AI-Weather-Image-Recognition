from sklearn.preprocessing import MultiLabelBinarizer

# Define the categories (weather classes)
categories = ["dew", "fogsmog", "frost", "glaze", "hail", "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"]

# Simulated multilabels for 10 images (e.g., rain with lightning, fog with frost, etc.)
multilabels = [
    ["rain", "lightning"],
    ["fogsmog", "frost"],
    ["hail"],
    ["rainbow", "rain"],
    ["sandstorm"],
    ["snow"],
    ["dew", "rime"],
    ["fogsmog"],
    ["lightning"],
    ["glaze", "frost"]
]

# Convert to binary vector representation
mlb = MultiLabelBinarizer(classes=categories)
binary_labels = mlb.fit_transform(multilabels)

# Display the binary labels
print("Multilabel Binarized Encoding:")
for i, label in enumerate(binary_labels):
    print(f"Image {i + 1}: {label}")