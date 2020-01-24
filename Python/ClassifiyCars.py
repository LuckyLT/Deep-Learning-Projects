import numpy as np
import os
import re
import pdb
import tensorflow as tf
from helper_func import *
from matplotlib.pyplot import imshow

# Data can be downloaded from https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder/data

path_data = os.path.abspath('Data/car_data')
path_train_data = os.path.join(path_data, 'train')
path_test_data = os.path.join(path_data, 'test')

train_filenames = []
test_filenames = []

def load_filenames(dir_path, list_filenames):
    for x in os.listdir(dir_path):
        folder_name = os.path.join(dir_path, x)
        list_filenames.extend([os.path.join(folder_name,x) for x in os.listdir(folder_name)])

load_filenames(dir_path=path_train_data, list_filenames=train_filenames)
load_filenames(dir_path=path_test_data, list_filenames=test_filenames)

len(train_filenames)
len(test_filenames)

for y, i in enumerate(os.listdir(path_train_data)):
    print(y)
    print(i, " : ", len([x for x in test_filenames if i in x]))

for y, i in enumerate(os.listdir(path_test_data)):
    print(y)
    print(i, " : ", len([x for x in test_filenames if i in x]))

test_labels = [re.search(r'(?<=test\\)[A-Za-z].+(?=\\)', x).group() for x in test_filenames]
train_labels = [re.search(r'(?<=train\\)[A-Za-z].+(?=\\)', x).group() for x in train_filenames]

len(train_labels)
len(test_labels)

train_filenames = np.array(train_filenames)
test_filenames = np.array(test_filenames)

# list_ds = tf.data.Dataset.list_files(path_train_data)

tst_resized = [tf.image.resize(tf.image.decode_image(tf.io.read_file(x)), (224, 224)) for x in train_filenames[:4]]
tst_raw = [tf.image.decode_image(tf.io.read_file(x)) for x in train_filenames[:4]]

tst_raw_cast = tf.cast(
    tst_raw[0],
    dtype='int16'
)

imshow(tst_raw[0])
imshow(tf.image.resize(tst_raw[0], (224, 224)))





train_filenames_images = [tf.image.decode_image(tf.io.read_file(x)) for x in train_filenames]


train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames_images, train_labels))
train_dataset = train_dataset.batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
for data, label in train_dataset:
    print(data, " with label : ", label)
    break

for data, label in train_dataset:
    print(data)
    print(label)
    break



# model.fit(dataset, steps_per_epoch = int(len(dataset) / 16(number epochs))

for i in os.listdir(path_train_data):
    print(i)

list_ds = tf.data.Dataset.list_files(path_train_data + '\\' + 'Volvo XC90 SUV 2007\\*')

for f in list_ds.take(1):
    tst = f



tst = list_ds.take(1)

def get_label(file_path):
  parts = tf.strings.split(file_path, '\\')
  return parts[-2] == 'Volvo XC90 SUV 2007'

label = tf.strings.split(tst, '\\')[-2]


IMG_WIDTH = 224
IMG_HEIGHT = 224

img = tf.io.read_file(tst)
img = tf.image.decode_jpeg(img, channels=3)  #color images
img = tf.image.convert_image_dtype(img, tf.float32)  #convert unit8 tensor to floats in the [0,1] range
imshow(tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]))




def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)  #color images
    img = tf.image.convert_image_dtype(img, tf.float32)  #convert unit8 tensor to floats in the [0,1] range
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) #resize the image into 224*224


def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  pdb.set_trace()
  img = decode_img(img)
  return img, label
AUTOTUNE=tf.data.experimental.AUTOTUNE

labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)