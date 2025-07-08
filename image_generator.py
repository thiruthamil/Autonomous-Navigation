import os
import json
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from PIL import Image # type: ignore

# Define paths
data_dir = 'D:\iddaw\IDDAW'

def get_all_files(directory, file_extension):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(file_extension):
                files.append(os.path.join(root, filename))
    return files

def load_image_paths_and_labels(base_dir):
    rgb_paths = []
    nir_paths = []
    labels = []

    conditions = ['fog', 'lowlight', 'rain', 'snow']
    for condition in conditions:
        condition_dir = os.path.join(base_dir, condition)
        if not os.path.exists(condition_dir):
            print(f"Directory missing for condition: {condition}")
            continue
        
        rgb_dir = os.path.join(condition_dir, 'rgb')
        nir_dir = os.path.join(condition_dir, 'nir')
        gtseg_dir = os.path.join(condition_dir, 'gtseg')

        for subdir in os.listdir(rgb_dir):
            rgb_subdir = os.path.join(rgb_dir, subdir)
            nir_subdir = os.path.join(nir_dir, subdir)
            gtseg_subdir = os.path.join(gtseg_dir, subdir)

            if os.path.isdir(rgb_subdir) and os.path.isdir(nir_subdir) and os.path.isdir(gtseg_subdir):
                rgb_paths.extend(get_all_files(rgb_subdir, '.png'))
                nir_paths.extend(get_all_files(nir_subdir, '.png'))

                label_files = get_all_files(gtseg_subdir, '.json')
                for json_path in label_files:
                    with open(json_path) as f:
                        data = json.load(f)
                        labels.append(data)

    return rgb_paths, nir_paths, labels

def image_generator(rgb_paths, nir_paths, labels, batch_size=32, time_steps=5):
    while True:
        for start in range(0, len(rgb_paths) - time_steps + 1, batch_size):
            end = min(start + batch_size, len(rgb_paths) - time_steps + 1)
            batch_rgb_sequences = [rgb_paths[i:i + time_steps] for i in range(start, end)]
            batch_nir_sequences = [nir_paths[i:i + time_steps] for i in range(start, end)]
            batch_labels = labels[start:end]

            batch_rgb_images = [[np.array(Image.open(path)) for path in seq] for seq in batch_rgb_sequences]
            batch_nir_images = [[np.array(Image.open(path).convert('L')) for path in seq] for seq in batch_nir_sequences]

            combined_sequences = []
            for rgb_seq, nir_seq in zip(batch_rgb_images, batch_nir_images):
                combined_seq = [np.concatenate((rgb, nir[..., np.newaxis]), axis=-1) for rgb, nir in zip(rgb_seq, nir_seq)]
                combined_sequences.append(combined_seq)

            # One-hot encode labels
            conditions = ['fog', 'lowlight', 'rain', 'snow']
            label_map = {condition: idx for idx, condition in enumerate(conditions)}
            batch_labels_encoded = [label_map[label['condition']] for label in batch_labels]
            batch_labels_encoded = tf.keras.utils.to_categorical(batch_labels_encoded, num_classes=4)

            yield np.array(combined_sequences), np.array(batch_labels_encoded)

# Load image paths and labels
train_rgb_paths, train_nir_paths, train_labels = load_image_paths_and_labels(os.path.join(data_dir, 'train'))
val_rgb_paths, val_nir_paths, val_labels = load_image_paths_and_labels(os.path.join(data_dir, 'val'))

# Ensure the data is not empty
if not train_rgb_paths or not val_rgb_paths or not train_nir_paths or not val_nir_paths:
    print("No training or validation images found.")
else:
    print(f"Number of training RGB samples: {len(train_rgb_paths)}")
    print(f"Number of training NIR samples: {len(train_nir_paths)}")
    print(f"Number of validation RGB samples: {len(val_rgb_paths)}")
    print(f"Number of validation NIR samples: {len(val_nir_paths)}")

    # Create data generators
    batch_size = 32
    time_steps = 5  # Number of frames in each sequence
    train_generator = image_generator(train_rgb_paths, train_nir_paths, train_labels, batch_size, time_steps)
    val_generator = image_generator(val_rgb_paths, val_nir_paths, val_labels, batch_size, time_steps)

    print("Data generators created successfully.")