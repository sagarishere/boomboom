import pandas as pd
import numpy as np
import face_recognition

# Function to load and filter the dataset, removing images without faces detected by HOG and CNN models
def load_filtered_data(filepath):
    df = pd.read_csv(filepath)
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

    filtered_images = []
    emotions = []
    indices_to_keep = []

    for index, row in df.iterrows():
        image_array = row['pixels'].reshape(48, 48).astype('uint8')
        # Check for faces using HOG model; if none found, check using CNN model
        if not face_recognition.face_locations(image_array, model="hog") and not face_recognition.face_locations(image_array, model="cnn"):
            indices_to_keep.append(index)
            filtered_images.append(' '.join(row['pixels'].astype(str)))  # Convert pixel values back to string
            emotions.append(row['emotion'])

    return filtered_images, emotions, indices_to_keep

print("Starting the data loading process with face detection filter...")
input_filepath = '../data/train.csv'
filtered_X, filtered_y, retained_indices = load_filtered_data(input_filepath)
print(f"Data loading completed. {len(filtered_X)} samples retained.")

# Construct a DataFrame with the filtered data
filtered_df = pd.DataFrame({
    'original_index': retained_indices,
    'emotion': filtered_y,
    'pixels': filtered_X
})

filtered_df.to_csv('../data/train_without_faces_cnn_hog.csv', index=False)
print("Filtered data saved to train_without_faces_cnn_hog.csv")

# Display basic info and a sample of the filtered DataFrame
filtered_df.info()
print("Filtered DataFrame Info displayed above.")
print(filtered_df.head())

# Remove duplicate entries from the original dataset
original_df = pd.read_csv('../data/train.csv')
print(f"Initial dataset size: {len(original_df)}")
duplicate_count = original_df.duplicated().sum()
print(f"Number of duplicate entries: {duplicate_count}")
unique_df = original_df.drop_duplicates()
unique_df.to_csv('../data/train_filtered.csv', index=False)
print(f"Dataset size after removing duplicates: {len(unique_df)}")