import tensorflow as tf
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
import shutil
import numpy as np
import pickle
from tensorflow.python.client import device_lib

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
device_lib.list_local_devices()


class MVGGNET:
    @staticmethod
    def build(height, width, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # first CONV => RELU => BN => CONV => RELU => BN => POOL => DO
        model.add(Rescaling(1./255))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # second CONV => RELU => BN => CONV => RELU => BN => POOL => DO
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

# Approach 1:
# Import the data as image_dataset_from_directory
#set relative path to data dir
data_dir = 'Data\\Fruit'
train_test_dir = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]

train_dir = train_test_dir[np.argmax([int(x.__contains__('Train')) for x in train_test_dir])]
test_dir = train_test_dir[np.argmax([int(x.__contains__('Test')) for x in train_test_dir])]
val_dir = train_test_dir[np.argmax([int(x.__contains__('Validation')) for x in train_test_dir])]

#set basic image processing variables
height, width = 64, 64
batch_size = 32

# training_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]

#1/4 of the training images that we have should be moved to the val folder at random
#1 test image per class into the test folder
# for i in training_files:
#     print('Folder: --> ', i)
#     all_jpg = [os.path.join(i, x) for x in os.listdir(i)]
#     random_jpg_test = list(np.random.choice(all_jpg, size=1, replace=False))
#     random_jpg_val = list(np.random.choice(all_jpg, size=int(len(all_jpg)/4), replace=False))
#     test_folder_random_jpg = list(np.random.choice([i.replace('Training', 'Test')], size=len(random_jpg_test)))
#     val_folder_random_jpg = list(np.random.choice([i.replace('Training', 'Validation')], size=len(random_jpg_test)))
#     os.makedirs(test_folder_random_jpg[0])
#     os.makedirs(val_folder_random_jpg[0])
#     for z, y in zip(random_jpg_test, test_folder_random_jpg):
#         print(z, '-->', y)
#         shutil.move(z, y)
#     for z, y in zip(random_jpg_test, val_folder_random_jpg):
#         print(z, '-->', y)
#         shutil.move(z, y)

#recreate the train_ds and val_ds
# create train set from directory
train_ds = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    shuffle=True,
    seed=123,
    image_size=(height, width),
    batch_size=batch_size)

# Found 49016 files belonging to 99 classes.
# Using 39213 files for training.
#create train validation from directory
val_ds = image_dataset_from_directory(
           val_dir,
           validation_split=0.2,
           subset="validation",
           seed=123,
           shuffle=True,
           image_size=(height, width),
           batch_size=batch_size)

#create train test from directory
test_ds = image_dataset_from_directory(
       test_dir,
       seed=123,
       shuffle=True,
       image_size=(height, width),
       batch_size=1)

#keep all the class names in a var
class_names = train_ds.class_names

with open('Fruit Classification//saved_models//miniVGGNET//assets//class_names.pickle', 'wb') as file:
    pickle.dump(class_names, file)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(24):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

#safe the fig
plt.savefig('Fruit Classification/Visualization/random_fruits_2.jpg')


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

#Config the data set performance
AUTOTUNE = tf.data.AUTOTUNE

#preserve the len of the classes
num_classes = len(train_ds.class_names)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#model checkpoint
checkpoint = ModelCheckpoint("./Fruit Classification/saved_models/Checkpoint",
                             monitor="val_loss", mode="min",
                             save_best_only=True,
                             verbose=1)

#Modelling
model = MVGGNET.build(width=32, height=32, depth=3, classes=num_classes)

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[checkpoint]
)


# the model has already been saved in the folder Checkpoint
# load the model
# model = tf.keras.models.load_model('Fruit Classification/saved_models/miniVGGNET')


#preserve the history
history = model.history

plt.figure()
plt.plot(range(len(history.epoch)), history.history['loss'], c='g', label='train loss')
plt.plot(range(len(history.epoch)), history.history['val_loss'], c='r', label='val loss')
plt.xticks(list(range(len(history.epoch))))
plt.legend()
plt.title('miniVGGNET Training vs Validation Loss')

#save the picture
plt.savefig('Fruit/Visualization/miniVGGNET.png')

#Comment: Overfitting is clear but let's evaluate the model and then we will seek how to prevent overfitting
model.evaluate(test_ds)
# loss: 0.1333 - accuracy: 0.9803 not bad

# loss: 0.1333 - accuracy: 0.9803 not bad

#take from test fruits folder and predict
for images, labels in test_ds.take(10):
    pred_data = np.argmax(model.predict(images), axis=1)
    prob_pred_data = np.max(model.predict(images), axis=1)
    for i in range(len(pred_data)):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Real: {test_ds.class_names[labels[i]]} \n Predicted: {class_names[pred_data[i]]} \n with probability {prob_pred_data[i]}")
        plt.axis("off")

#results are ridiculous, but the algorithm hasn't seen them
plt.savefig("Fruit Classification/Visualization/Multiple_Fruits.png")

import cv2
d = cv2.imread('D:\Projects\\01.DeepLearning\Deep-Learning-Projects\Data\Fruit\Test\Hazelnut\\r_91_100.jpg')
d.shape
cv2.imshow('Image', d)

d_r = cv2.resize(d, (64, 64))
d_r = np.expand_dims(d_r, axis=0)
d_s = d_r/255.
cv2.imshow('Image', d_s)

class_names[np.argmax(model.predict(d_s))]

