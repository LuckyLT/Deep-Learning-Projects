import os
import numpy as np
import shutil
import cv2
import imutils

def split_data_dir(data_dir):
    """
    The function splits the data in Train, Test and Validation folders
    Prerequsites: Train folder with all the images
    """
    for x in ["Test", "Validation"]:
        if x not in os.listdir(data_dir):
            print(f'Creating {x} folder')
            os.makedirs(os.path.join(data_dir, x))
        else:
            print(f"{x} folder exists!")

    file_dirs = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    train_dir = file_dirs[np.argmax([int(x.__contains__('Train')) for x in file_dirs])]
    training_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]

    for i in training_files:
        print('Folder: --> ', i)
        all_jpg = [os.path.join(i, x) for x in os.listdir(i)]
        random_jpg_test = list(np.random.choice(all_jpg, size=1, replace=False))
        random_jpg_val = list(np.random.choice([x for x in all_jpg if x != random_jpg_test[0]], size=int(len(all_jpg)/4), replace=False))
        test_folder_random_jpg = list(np.random.choice([i.replace('Train', 'Test')], size=len(random_jpg_test)))
        val_folder_random_jpg = list(np.random.choice([i.replace('Train', 'Validation')], size=len(random_jpg_val)))
        try:
            os.makedirs(test_folder_random_jpg[0])
        except:
            print(f'Folder {test_folder_random_jpg[0]} already exist!')
        try:
            os.makedirs(val_folder_random_jpg[0])
        except:
            print(f'Folder {val_folder_random_jpg[0]} already exist!')

        for z, y in zip(random_jpg_test, test_folder_random_jpg):
            print(z, '-->', y)
            shutil.move(z, y)
        for z, y in zip(random_jpg_val, val_folder_random_jpg):
            print(z, '-->', y)
            shutil.move(z, y)

class AspectPreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # grab the dimensions of the image and then initialize the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # if the width is smaller than the height, then resize
        # along the width (i.e., the smaller dimension) and then
        # update the deltas to crop the height to the desired dimension
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)

        # otherwise, the height is smaller than the width so
        # resize along the height and then update the deltas to crop along the width

        else:
            image = imutils.resize(image, height=self.height, inter=self.height)
            dW = int((image.shape[1] - self.width) / 2.0)

        # now that our images have been resized, we need to re-grab the width and height, followed by performing the crop
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        # finally, resize the image to the provided spatial dimensions
        # to ensure our output image is always a fixed size

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)