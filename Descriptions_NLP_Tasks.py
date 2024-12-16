# Define a function to label categories
def label_category(category):
    categories = {"electronics": 0, "books": 1, "clothing": 2}
    return categories.get(category.lower(), -1)

# Apply labeling
df['category_label'] = df['category'].apply(label_category)

# Filter valid labels
df = df[df['category_label'] != -1]

print(df['category_label'].value_counts())