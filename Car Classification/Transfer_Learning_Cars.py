import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout, Dense, Flatten, Input
from tensorflow.keras import Model
from helper_functions import *

# Data can be downloaded from https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder/data

# Define some variables
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32

#define data dir
data_dir = 'Data/Cars/'

#call the split data dir function defined in helper_functions.py
# split_data_dir(data_dir)

#build the model
model = ResNet50(weights='imagenet', include_top=False)

#explore the model
for i, layer in enumerate(model.layers):
    print(f"{i}, \t {layer.__class__.__name__}")

class CarNN:
    @staticmethod
    def build(baseModel, classes, D):
        headModel = baseModel.output
        headModel = Flatten(name='flatten')(headModel)
        headModel = Dense(D, activation='relu')(headModel)
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(classes, activation='softmax')(headModel)
        return headModel


aug_train = ImageDataGenerator(rescale=1./255,
                               rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

aug_test = ImageDataGenerator(rescale=1./255)

train_gen = aug_train.flow_from_directory('./Data/Cars/Train',
                                          seed=69,
                                          target_size=(IMG_WIDTH, IMG_HEIGHT),
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

val_gen = aug_test.flow_from_directory('./Data/Cars/Validation',
                                          seed=69,
                                          target_size=(IMG_WIDTH, IMG_HEIGHT),
                                          batch_size=BATCH_SIZE)

baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
headModel = CarNN.build(baseModel, train_gen.num_classes, 256)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_gen, validation_data=val_gen, epochs=4, steps_per_epoch=int(7838/32))

