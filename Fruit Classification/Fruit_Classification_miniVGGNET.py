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
import shutil
import numpy as np
from PIL import Image



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
val_dir = train_test_dir[np.argmax([int(x.__contains__('Test')) for x in train_test_dir])]
test_dir = train_test_dir[np.argmax([int(x.__contains__('Multiple')) for x in train_test_dir])]


#all the classes from val in train which is ok
val_classes_to_be_moved = [os.path.join(test_dir, x) for x in val_ds.class_names if x not in train_ds.class_names]
train_classes_to_be_moved = [os.path.join(train_dir, x) for x in val_ds.class_names if x not in train_ds.class_names]

#move some of the images in train
for i, y in zip(val_classes_to_be_moved, train_classes_to_be_moved):
    print(i, '-->', y)
    shutil.move(i, y)

#1/4 of the training images that we moved should go back to the val folder at random
for i in train_classes_to_be_moved:
    print('Folder: --> ', i)
    all_jpg = [os.path.join(i, x) for x in os.listdir(i)]
    random_jpg = list(np.random.choice(all_jpg, size=int(len(all_jpg)/4), replace=False))
    val_folder_random_jpb = list(np.random.choice([i.replace('Training', 'Test')], size=len(random_jpg)))
    os.makedirs(val_folder_random_jpb[0])
    for z, y in zip(random_jpg, val_folder_random_jpb):
        print(z, '-->', y)
        shutil.move(z, y)

#set basic image processing variables
height, width = 64, 64
batch_size = 32

# create train set from directory
train_ds = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
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
           image_size=(height, width),
           batch_size=batch_size)

#create train test from directory
test_ds = image_dataset_from_directory(
       test_dir,
       seed=123,
       image_size=(height, width),
       batch_size=1)

#keep all the class names in a var
class_names = train_ds.class_names

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

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(train_ds.class_names)

#Modelling
model = MVGGNET.build(width=32, height=32, depth=3, classes=num_classes)
model.summary()

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10
)

#save the model
model.save('Fruit Classification/saved_models/miniVGGNET')

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
model.evaluate(val_ds)
# loss: 0.1333 - accuracy: 0.9803 not bad

pred_data = np.argmax(model.predict(test_ds), axis=1)
list(set([class_names[x] for x in pred_data])) == test_ds.class_names

# loss: 0.1333 - accuracy: 0.9803 not bad

#predict random fruits
for images, labels in val_ds.take(5):
    pred_data = np.argmax(model.predict(images), axis=1)
    for i in range(len(pred_data)):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Real: {class_names[labels[i]]} \n Predicted: {class_names[pred_data[i]]}")
        plt.axis("off")

    plt.savefig("Fruit Classification/Visualization/Fruits.png")

#take from multiple fruits folder and predict
for images, labels in test_ds.take(1):
    pred_data = np.argmax(model.predict(images), axis=1)
    prob_pred_data = np.max(model.predict(images), axis=1)
    for i in range(len(pred_data)):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Real: {test_ds.class_names[labels[i]]} \n Predicted: {class_names[pred_data[i]]} \n with probability {prob_pred_data[i]}")
        plt.axis("off")

#results are ridiculous, but the algorithm hasn't seen them
plt.savefig("Fruit Classification/Visualization/Multiple_Fruits.png")