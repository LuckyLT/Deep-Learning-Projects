import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from skimage.io import imread
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
import pdb

# Constants
TRAIN_PCT, VAL_PCT, TEST_PCT = 0.9, 0.05, 0.05

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


DATA_DIR = os.path.join(os.getcwd(), 'Fruits')
os.listdir(DATA_DIR)

TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'Test')

fruits = os.listdir(TRAIN_DATA_DIR)
trainLabels = [fruits[fruits.index('Avocado')]] + [fruits[fruits.index('Banana')]]


def get_all_filenames(base_dir, class_names_plural=True):
    """
    Returns the filenames and their corresponding classes.
    Assumes the following structure:
    |---base_dir
    | |---class1
    | | |---image1.jpg
    | | |---image2.jpg
    | |---class2
    | | |---image1.jpg
    | | |---image2.jpg

    Since in our case we can infer the class from the filename,
    we could skip returning it but we'll do so for clarity.
    """
    filenames = {
        "image_filename": [],
        "image_class": []
    }

    for image_class in os.listdir(base_dir):
        # pdb.set_trace()
        image_class_dir = os.path.join(base_dir, image_class)
        image_class_dir_files = os.listdir(image_class_dir)

        filenames_in_class = [os.path.join(image_class_dir, file) for file in image_class_dir_files]
        filenames["image_filename"].extend(filenames_in_class)

        normalized_image_class = image_class
        filenames["image_class"].extend([normalized_image_class] * len(filenames_in_class))
    return pd.DataFrame(filenames)


filenames = get_all_filenames(TRAIN_DATA_DIR)
filenames = filenames[filenames.image_class.isin(trainLabels)]

def get_image_dimensions(image_filename):
    """
    Returns the dimensions of the image (height, width, channels) in pixels.

    There are better methods which don't involve reading the entire image
    and loading it in memory but this is simple enough.
    """
    return imread(image_filename).shape

dims = filenames.image_filename.map(get_image_dimensions)
dimensions = pd.DataFrame([*dims], columns=['height', 'width', 'channels'])

filenames = pd.merge(filenames.reset_index(drop=True), dimensions, left_index=True, right_index=True)

filenames.describe()

# SPLIT the data

train_data, val_data, test_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

for class_name, data in filenames.groupby("image_class"):
    # Randomize the order (shuffle) before splitting
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

# Preparing images for modelling
def read_and_prepare_image(image_filename, image_class):
    # Get image
    # pdb.set_trace()
    image = tf.io.read_file(image_filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, IMAGE_SIZE)

    image_class_encoded = tf.where(image_class == 'Avocado', 1, 0)
    return image, image_class_encoded

def initialize_tf_dataset(data, should_batch=True, should_repeat=True):
    # pdb.set_trace()
    dataset = tf.data.Dataset.from_tensor_slices((data.image_filename.values, data.image_class.values))
    dataset = dataset.map(read_and_prepare_image)
    dataset = dataset.shuffle(buffer_size=len(data))

    if should_batch:
        dataset = dataset.batch(BATCH_SIZE)
    else:
        dataset = dataset.batch(len(data))

    if should_repeat:
        dataset = dataset.repeat()

    return dataset


train_data = initialize_tf_dataset(train_data)
val_data = initialize_tf_dataset(val_data)
test_data = initialize_tf_dataset(test_data)


# "Getting a model for transfer learning"
resnet50 = ResNet50()
resnet50.summary()
resnet50_conv = Model(inputs=resnet50.get_layer("input_1").input, outputs=resnet50.get_layer("avg_pool").output)
resnet50_conv.summary()
# Sequential makes the usage a bit simpler. # Also, adding the resnet
# separately allows us to see a shorter summary
model = Sequential()
model.add(resnet50_conv)
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()
model.get_layer("model").trainable = False

model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=[BinaryAccuracy()])


steps_per_epoch_train = round(len(filenames) * TRAIN_PCT / BATCH_SIZE)
steps_per_epoch_val = round(len(filenames) * VAL_PCT / BATCH_SIZE)

steps_per_epoch_train, steps_per_epoch_val

history = model.fit(
    train_data,
    epochs=1,
    steps_per_epoch=steps_per_epoch_train,
    validation_data=val_data,
    validation_steps=steps_per_epoch_val)


for i in train_data:
    tst = i[0]
    lbl = i[1]
    break

model.evaluate(train_data, steps=5)
