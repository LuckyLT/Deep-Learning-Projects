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

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train, x_test = x_train/255.0, x_test/255.0

model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),
               loss=SparseCategoricalCrossentropy())

model.fit(x_train, y_train, epochs=2)

model.evaluate(x_train, y_train, verbose=1)
model.evaluate(x_test, y_test, verbose=1)