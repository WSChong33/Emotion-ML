# Coverting CSV files (image data + emotion labels) -> PNG files + saved in dataset dir

import os
import pandas as pd
import numpy as np
import cv2

# Function to make new program
def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to create dataset directory in the parent directory of the current working directory
def create_parent_dir_if_not_exists(dir_name):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    new_dir_path = os.path.join(parent_dir, dir_name)
    create_dir_if_not_exists(new_dir_path)
    return new_dir_path

# Function to convert CSV file (each row) into PNG files for model training
def save_images_from_csv(csv_path, output_dir):
    data = pd.read_csv(csv_path) # Read into a dataframe (table)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    for emotion in emotions:
        create_dir_if_not_exists(os.path.join(output_dir, 'train', emotion))
        create_dir_if_not_exists(os.path.join(output_dir, 'test', emotion))

    for index, row in data.iterrows():
        emotion = emotions[row['emotion']]
        pixels = np.array(row['pixels'].split(), dtype='uint8').reshape(48, 48)

        usage = row['Usage']
        if usage == 'Training':
            usage_dir = 'train'
        else:
            usage_dir = 'test'

        image_path = os.path.join(output_dir, usage_dir, emotion, f"{index}.png")
        cv2.imwrite(image_path, pixels)

if __name__ == "__main__":
    dataset_dir = create_parent_dir_if_not_exists('dataset')
    
    csv_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset.csv')
    save_images_from_csv(csv_path, dataset_dir)
