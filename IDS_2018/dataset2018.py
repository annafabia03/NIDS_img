import pandas as pd
import numpy as np
import os
from PIL import Image




df=pd.read_csv('/Users/annafabia/Desktop/NIDS_deep_images/IDS_2018/IDS2018data.csv')


# Separate benign and malicious samples
benign_samples = df[df.iloc[:, -1] == "BENIGN"]
malicious_samples = df[df.iloc[:, -1] != "BENIGN"]

# Sample 15,000 from each group
benign_sampled = benign_samples.sample(n=30000, random_state=42)
malicious_sampled = malicious_samples.sample(n=30000, random_state=42)

# Combine the samples
sampled_df = pd.concat([benign_sampled, malicious_sampled])

labels = sampled_df.iloc[:, -1].values
features = sampled_df.iloc[:, :-1].values

# Count the number of 'BENIGN' and malicious labels
num_benign = np.sum(labels == "BENIGN")
num_malicious = len(labels) - num_benign

print(f'Number of BENIGN samples: {num_benign}')
print(f'Number of malicious samples: {num_malicious}')

# Define the dimensions of the grayscale image
image_shape = (7, 10)

# Normalize the features to range [0, 255]
def normalize_features(features):
    min_val = np.min(features)
    max_val = np.max(features)
    
    if min_val == max_val or np.isinf(min_val) or np.isinf(max_val):
        # Handle cases where all values are the same or infinite
        return np.zeros_like(features)
    else:
        normalized = (features - min_val) / (max_val - min_val) * 255
        return normalized.astype(np.uint8)

# Convert each row to a grayscale image
images = []
for feature_row in features:
    # Normalize the feature values to be between 0 and 255
    normalized_row = normalize_features(feature_row)
    
    # Reshape to the desired image shape
    image_array = normalized_row.reshape(image_shape)
    
    # Convert to PIL Image
    image = Image.fromarray(image_array, mode='L')
    
    # Append image to list
    images.append(image)

# Save images to disk
def save_images(images, labels, directory):
    os.makedirs(directory, exist_ok=True)
    for idx, (image, label) in enumerate(zip(images, labels)):
        label_int = 0 if label == "BENIGN" else 1
        image_filename = os.path.join(directory, f'image_{idx}_{label_int}_{label}.png')
        image.save(image_filename)

# Save sampled images
save_images(images, labels, 'IDS2018_images')

print('Image conversion and saving completed.')
