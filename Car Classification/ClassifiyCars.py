import numpy as np
import pandas as pd
import os
import re
import pdb
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imread
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy, AUC, SparseCategoricalAccuracy
import shutil
from functions import *

# Data can be downloaded from https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder/data

with open('pandasOptions.env', 'r') as f:
    for line in f:
        pd.set_option(line.split("=")[0], int(line.split("=")[1]))

# Define some useful functions
IMG_WIDTH, IMG_HEIGHT = 224, 224

TRAIN_PCT, VAL_PCT, TEST_PCT = 0.9, 0.05, 0.05

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# define file paths
path_data = os.path.abspath('Data/car_data')
train_path_data = os.path.join(path_data, 'train')
SPLITS_DIR = "Data\\car_data\\splits"

labels = os.listdir(train_path_data)

# TODO fix filepath

Filenames = load_filenames(dir_path=train_path_data)

# Data exploration
Filenames.groupby('image_class').size()  # we might need over sampling

Filenames.shape  # (10496, 2)

dims = Filenames.image_file.map(get_image_dimensions)

dims = pd.DataFrame([*dims], columns=["height", "width", "channels"])
len(dims[dims.channels != 3])  # 20 gray images

# exclude the bad images
Filenames = Filenames.join(dims).loc[dims.channels == 3]

# Distribution of image dimensions
fig, (h_ax, w_ax, res_ax) = plt.subplots(1, 3, figsize=(18, 6))
h_ax.hist(dims.height, bins=30)
h_ax.set_xlabel("Image height")

w_ax.hist(dims.width, bins=30)
w_ax.set_xlabel("Image width")

res_ax.hist(dims.width / dims.height, bins=40)
res_ax.set_xlabel("Aspect ratio")

plt.suptitle("Distributions of image dimensions")
plt.show()

# Split the data
train_data, val_data, test_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

for class_name, data in Filenames.groupby("image_class"):
    # Randomize the order (shuffle) before splitting
    # pdb.set_trace()
    data = data.sample(frac=1)
    train_end_index, val_end_index = round(TRAIN_PCT * len(data)), round((TRAIN_PCT + VAL_PCT) * len(data))
    train_data_in_class, val_data_in_class, test_data_in_class = \
        data[:train_end_index], data[train_end_index: val_end_index], data[val_end_index:]
    train_data = pd.concat([train_data, train_data_in_class])
    val_data = pd.concat([val_data, val_data_in_class])
    test_data = pd.concat([test_data, test_data_in_class])

# Randomize the sets once again so the classes are not consecutive
train_data = train_data.sample(frac=1)
val_data = val_data.sample(frac=1)
test_data = val_data.sample(frac=1)

if not os.path.exists(SPLITS_DIR):
    os.makedirs(SPLITS_DIR)

for dataset, filename in zip([train_data, val_data, test_data], ["train", "val", "test"]):
    dataset.to_csv(os.path.join(SPLITS_DIR, filename + ".csv"), index=False)

# Load the dataframes
train_data = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
val_data = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
test_data = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))


def generate_one_hot_labels(labels, df):
    raw_labels = pd.Series(labels, name='image_class').reset_index()
    index_order = pd.merge(df, raw_labels, how='left', on='image_class')['index']
    labels_one_hot_encoded = tf.one_hot([*index_order], depth=len(labels))  # works
    return(labels_one_hot_encoded)

labels_train_one_hot_encoded = generate_one_hot_labels(labels, train_data)
labels_val_one_hot_encoded = generate_one_hot_labels(labels, val_data)
labels_test_one_hot_encoded = generate_one_hot_labels(labels, test_data)

labels_train = tf.argmax(labels_train_one_hot_encoded, axis=1)
labels_val = tf.argmax(labels_val_one_hot_encoded, axis=1)
labels_test = tf.argmax(labels_test_one_hot_encoded, axis=1)

train_data = initialize_tf_dataset(train_data, labels_train)
val_data = initialize_tf_dataset(val_data, labels_val)
test_data = initialize_tf_dataset(test_data, labels_test, should_batch=False, should_repeat=False)

for i in train_data.take(1):
    tst = i[0]


# my approach
# train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, image_class_encoded))
# train_dataset = train_dataset.map(load_images)
# train_dataset = train_dataset.shuffle(len(train_filenames))
# train_dataset = train_dataset.batch(32)
# train_dataset = train_dataset.repeat()
#
# test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
# test_dataset = test_dataset.map(load_images)
# test_dataset = test_dataset.shuffle(len(train_filenames))
# test_dataset = test_dataset.batch(32)
# test_dataset = test_dataset.repeat()
#
# image_batch, label_batch = next(iter(train_dataset))
# imshow(image_batch[0])
# label_batch[0]

babyNN = Sequential([
    Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    Conv2D(20, 3, padding="same", activation='relu'),
    Conv2D(20, 3, padding="same", activation='relu'),
    MaxPool2D(),
    Conv2D(80, 3, padding="same", activation='relu'),
    Conv2D(80, 3, padding="same", activation='relu'),
    MaxPool2D(),
    BatchNormalization(),
    Conv2D(140, 3, padding="same", activation='relu'),
    Conv2D(140, 3, padding="same", activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(300, activation='relu'),
    Dense(48, activation='softmax')
])

babyNN.summary()

babyNN.compile(optimizer=Adam(learning_rate=0.001),
               loss=SparseCategoricalCrossentropy())

# TRAINING
steps_per_epoch_train = round(len(Filenames.image_file.values) * TRAIN_PCT / BATCH_SIZE)
steps_per_epoch_val = round(len(Filenames.image_file.values) * VAL_PCT / BATCH_SIZE)

steps_per_epoch_train, steps_per_epoch_val

history = babyNN.fit(train_data,
                     epochs=1,
                     steps_per_epoch=steps_per_epoch_train,
                     validation_data=val_data,
                     validation_steps=steps_per_epoch_val)


tf.argmax(labels_train_one_hot_encoded, axis=1)