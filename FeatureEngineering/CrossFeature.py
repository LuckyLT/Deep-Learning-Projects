# Feature Cross
# a synthetic feature formed by multiplying two or more features
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import feature_column as fc
import pandas as pd

data = pd.read_csv('../Data/Embeddings/babyweight_sample.txt')
# data.shape --> (999, 5)


#feature cross
gender_x_plurality = fc.crossed_column(['is_male', 'plurality'], hash_bucket_size=1000)
crossed_feature = fc.embedding_column(gender_x_plurality, dimension=2)
crossed_feature = fc.indicator_column(gender_x_plurality)


