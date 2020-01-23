import numpy as np
import os
import re
import pdb
import tensorflow as tf
from helper_func import *

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

list_ds = tf.data.Dataset.list_files(path_train_data)


for f in list_ds.take(1):
  print(f)



train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_dataset = train_dataset.map(load_image).shuffle(len(train_filenames))
train_dataset = train_dataset.batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
for data, label in train_dataset:
    print(data, " with label : ", label)
    break

for data, label in test_dataset:
    print(data)
    print(label)
    break

t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])

t.numpy().shape

# model.fit(dataset, steps_per_epoch = int(len(dataset) / 16(number epochs))