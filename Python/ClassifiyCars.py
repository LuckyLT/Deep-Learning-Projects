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

def load_filenames(dir_path):
    """
            Returns the filenames and their corresponding classes.
    """

    filenames = {
        'image_file': [],
        'image_class': []
    }

    for car_brand in os.listdir(dir_path):
        folder_name = os.path.join(dir_path, car_brand)

        filenames_in_class = [os.path.join(folder_name, x) for x in os.listdir(folder_name)]
        filenames['image_file'].extend(filenames_in_class)

        filenames["image_class"].extend([car_brand] * len(filenames_in_class))


    return(pd.DataFrame(filenames))

# TODO fix filepath

Filenames = load_filenames(dir_path=train_path_data)

# Data exploration
Filenames.groupby('image_class').size() #we might need over sampling

Filenames.shape #(10496, 2)

def get_image_dimensions(image_filename):
    """
    Returns the dimensions of the image (height, width, channels) in pixels.

    There are better methods which don't involve reading the entire image
    and loading it in memory but this is simple enough.
    """
    return imread(image_filename).shape

dims = Filenames.image_file.map(get_image_dimensions)

dims = pd.DataFrame([*dims], columns=["height", "width", "channels"])
len(dims[dims.channels != 3]) #20 gray images

#exclude the bad images
Filenames = Filenames.join(dims).loc[dims.channels == 3]

#Distribution of image dimensions
fig, (h_ax, w_ax, res_ax) = plt.subplots(1, 3, figsize = (18, 6))
h_ax.hist(dims.height, bins = 30)
h_ax.set_xlabel("Image height")

w_ax.hist(dims.width, bins = 30)
w_ax.set_xlabel("Image width")

res_ax.hist(dims.width / dims.height, bins = 40)
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

def load_images(image_filename, image_label):

    # read the file and then decode it
    result_file = tf.io.read_file(image_filename)
    result_image = tf.image.decode_jpeg(result_file) #decode_jpeg

    # result_image = tf.image.convert_image_dtype(result_image, tf.float32) #convert to float32

    # resize the image
    result_image = tf.image.resize(result_image, (224, 224)) #resize the data

    def preprocess_image(x):
        """
        This is a stripped-down version of Keras' own imagenet preprocessing function,
        as the original one is throwing an exception
        """
        pdb.set_trace()
        backend = tf.keras.backend

        # 'RGB'->'BGR'
        x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

        mean_tensor = backend.constant(-np.array(mean))

        # Zero-center by mean pixel
        if backend.dtype(x) != backend.dtype(mean_tensor):
            x = backend.bias_add(
                x, backend.cast(mean_tensor, backend.dtype(x)))
        else:
            x = backend.bias_add(x, mean_tensor)
        if std is not None:
            x /= std
        return x

    image = preprocess_image(result_image)

    return image

    # Return the correct class
image_class_encoded = tf.one_hot(labels, depth=len(set(labels)))

train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_dataset = train_dataset.map(load_images)
train_dataset = train_dataset.shuffle(len(train_filenames))
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test_dataset = test_dataset.map(load_images)
test_dataset = test_dataset.shuffle(len(train_filenames))
test_dataset = test_dataset.batch(32)
test_dataset = test_dataset.repeat()


image_batch, label_batch = next(iter(train_dataset))
imshow(image_batch[0])
label_batch[0]


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
    Dense(400, activation='relu'),
    Dense(300, activation='relu'),
    Dense(196, activation='softmax')
])


babyNN.summary()

babyNN.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy()])

history = babyNN.fit(train_dataset,
    steps_per_epoch=int(len(train_filenames) / 32)
)



