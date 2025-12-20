import os

import numpy as np
import pandas as pd

PATH = "../data"
TRAIN_PATH = os.path.join(PATH, 'train.csv')
VAL_PATH = os.path.join(PATH, 'validation.csv')
TEST_PATH = os.path.join(PATH, 'test.csv')
SAMPLES_PATH = os.path.join(PATH, 'samples')

def load_image_filenames():
    filenames = []
    for filename in os.listdir(SAMPLES_PATH):
        if filename.endswith(".npy"):
            filenames.append(filename.split(".npy")[0])
    return pd.DataFrame(filenames, columns=['id'])

def load_image(filename):
    path = os.path.join(SAMPLES_PATH, filename + '.npy')
    image = np.load(path)
    image = np.astype(image, np.float32)
    return image

def load_data(subset):
    df = pd.DataFrame()
    if subset == "train":
        df = pd.read_csv(TRAIN_PATH)
    elif subset == "val":
        df = pd.read_csv(VAL_PATH)
    elif subset == "test":
        df = pd.read_csv(TEST_PATH)
    return df

#
# def load_samples():
#     samples = []
#     print(f"Loading samples and extracting features from: {SAMPLES_PATH}")
#
#     for filename in tqdm(os.listdir(SAMPLES_PATH)):
#         if filename.endswith('.npy'):
#             image_id = os.path.splitext(filename)[0]
#             image_path_full = os.path.join(SAMPLES_PATH, filename)
#
#             try:
#                 image = np.load(image_path_full)
#                 image = image.astype(np.float32)
#                 samples.append({"sample":image, "id":image_id})
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")
#
#     df_samples = pd.DataFrame(samples)
#     df_samples.set_index('id', inplace=True)
#     return df_samples
#
# def load():
#     df_train = pd.read_csv(TRAIN_PATH)
#     print(f"Shape of df_train: {df_train.shape}")
#     df_val = pd.read_csv(VAL_PATH)
#     print(f"Shape of df_val: {df_val.shape}")
#     df_test = pd.read_csv(TEST_PATH)
#     print(f"Shape of df_test: {df_test.shape}")
#     df_samples = load_samples()
#     print(f"\nShape of df_samples: {df_samples.shape}")
#     return {
#         "samples": df_samples,
#         "train": df_train,
#         "val": df_val,
#         "test": df_test
#     }
#
# data = load()