from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from helper_functions import *

# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension

print('Loading image...')
path = 'Data/Cars/Acura/00002.jpg'
image = load_img(path)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

ap = AspectPreprocessor(224, 224)
ap.preprocess(image)

#initialize ImageDataGenerator
aug = ImageDataGenerator(rotation_range=30, #degree range of the random rotations +-30 degrees
                         width_shift_range=0.1,  #horizontal shift (fraction of the given dimension 10%)
                         height_shift_range=0.1, #vertical shift (fraction of the given dimension 10%)
                         shear_range=0.2, #controls the angle in counterclockwise direction as radians
                         zoom_range=0.2,# a floating point value that allows the image to be “zoomed in” or “zoomed out” according to the following uniform distribution of values: [1 - zoom_range, 1 + zoom_range]
                         horizontal_flip=True, #flipped horizontally during the training process
                         fill_mode='nearest')


print("Generating images...")
imageGen = aug.flow(image, batch_size=2, save_to_dir='Data/Output/', save_prefix='output', save_format='jpg')

total = 0
#exploring the data generator
for image in imageGen:
    total += 1
    if total == 10:
        break


