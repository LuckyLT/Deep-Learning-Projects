import numpy as np
import pandas as pd
import os
import re
import pdb
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imread
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy, AUC, SparseCategoricalAccuracy

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train, x_test = x_train/255.0, x_test/255.0

model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.fit(x_train, y_train, epochs=4)

model.evaluate(x_test, y_test, verbose=0)