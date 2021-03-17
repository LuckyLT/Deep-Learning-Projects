#An Embedding is a low-dimensional, vector representation of a high dimensional feature which maintains the semantic meaning
#of the feature in a such a way that similar features are close in the embedding space

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow.feature_column as fc

# Embedding Layer for categorical variable
data = pd.read_csv('../Data/Embeddings/babyweight_sample.txt')
data.plurality.head(6)
data.plurality.unique()

#define classes with its retrospective indexes
CLASSES = {key: value for key, value in zip(data.plurality.unique(), range(len(data.plurality.unique())))}
N_CLASSES = len(CLASSES)

#convert to a numeric index
plurality_index = [CLASSES[p] for p in data.plurality]

EMBED_DIM = 2

embedding_layer_1 = layers.Embedding(input_dim=N_CLASSES,
                                   output_dim=EMBED_DIM)

embedding_layer_2 = embedding_layer_1(tf.constant(plurality_index))
#print the embedding_layer_2
embedding_layer_2[:5]

#scatter plot the first and the second column
plt.scatter(embedding_layer_2[:, 0], embedding_layer_2[:, 1])