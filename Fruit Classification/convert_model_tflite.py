import tensorflow as tf
import os

#define the model dir from where we will convert the model to .tflite
model_dir = os.getcwd() + '\\Fruit Classification\\saved_models\\Checkpoint\\'
#define the converter
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)

#convert to tflite
tflite_model = converter.convert()

#open io file and write
with tf.io.gfile.GFile(os.getcwd() + '\\Fruit Classification\\saved_models\\tflite\\miniVGGNET.tflite', 'wb') as f:
    f.write(tflite_model)

