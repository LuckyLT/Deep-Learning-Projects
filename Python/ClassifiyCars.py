import numpy as np
import os
import re
import pdb
import tensorflow as tf
from matplotlib.pyplot import imshow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy, AUC, SparseCategoricalAccuracy

# Data can be downloaded from https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder/data

# Define some useful functions
IMG_WIDTH = 224
IMG_HEIGHT = 224

def load_filenames(dir_path, list_filenames):
    """ load the full image names in a list """
    for x in os.listdir(dir_path):
        folder_name = os.path.join(dir_path, x)
        list_filenames.extend([os.path.join(folder_name, x) for x in os.listdir(folder_name)])

# TODO fix filepath
def load_images(filename, label):

    result_file = tf.io.read_file(filename)
    result_image = tf.image.decode_jpeg(result_file) #decode_jpeg
    result_image = tf.image.convert_image_dtype(result_image, tf.float32) #convert to float32
    result_image = tf.image.resize(result_image, (224, 224)) #resize the data
    return (result_image, label)

def get_label(file_path):
    parts = tf.strings.split(file_path, '\\')
    return parts[-2]

def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3) #color images
  img = tf.image.convert_image_dtype(img, tf.float32) #convert unit8 tensor to floats in the [0,1]range
  img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) #resize the image into 224*224
  return img

def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# define file paths
path_data = os.path.abspath('Data/car_data')
path_train_data = os.path.join(path_data, 'train')
path_test_data = os.path.join(path_data, 'test')

train_filenames = []
test_filenames = []

load_filenames(dir_path=path_train_data, list_filenames=train_filenames)
load_filenames(dir_path=path_test_data, list_filenames=test_filenames)

print(len(train_filenames))  # 8144
print(len(test_filenames))  # 8041

# count the number of train images in each folder
for i, j in zip(os.listdir(path_train_data), os.listdir(path_train_data)):
    # print(y)
    print(i, "- train: ", len([x for x in test_filenames if i in x])," test: ", len([x for x in test_filenames if i in x]))

# extract the labels for training and test set
# TODO could be integrated in the function load_filenames
train_labels = [re.search(r'(?<=train\\)[A-Za-z].+(?=\\)', x).group() for x in train_filenames]
test_labels = [re.search(r'(?<=test\\)[A-Za-z].+(?=\\)', x).group() for x in test_filenames]

print(len(train_labels))
print(len(test_labels))

train_filenames = np.array(train_filenames)
test_filenames = np.array(test_filenames)

train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

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



