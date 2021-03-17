import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Embedding Layers in Keras
#Load the dataset
log_dir = './logs'
data_dir = '../Data/Embeddings/'

data = 'titles_full.csv'
data_path = os.path.join(data_dir, data)
cols = ['title', 'source']

titles_df = pd.read_csv(data_path, names=cols)
titles_df = titles_df.sample(frac=1)
titles_df.head()

#keras Tokenizer will help us with some exploratory text statistics - like:
# number of words in our dataset
# how many titles we have
# max len of titles we have

#invoke the Tokenizer class from keras
tokenizer = Tokenizer()
#fit on texts will update the internal vocabulary based on a list of texts
tokenizer.fit_on_texts(titles_df.title)

#transforms each text in texts to a sequence of integers
integerized_titles = tokenizer.texts_to_sequences(titles_df.title)

#check the first 5 integerized titles
integerized_titles[:5]

#get the len of the full word index, dataset size and max len title
vocabulary_size = len(tokenizer.word_index)
dataset_size = tokenizer.document_count
max_len = max([len(x) for x in integerized_titles])

#Preprocess the data
# Define a function which will pad the elements for our titles to feed them into the model
def pad_titles(texts, MAX_len=max_len):
    sequence = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequence, MAX_len, padding='post')
    return padded_sequences

#classes as numerical representation
CLASSES = {y: x for x, y in enumerate(np.unique(titles_df.source))}

#len of the classes
N_CLASSES = len(CLASSES)

#create a function which will one hot encode the labels
def encode_labels(sources):
    classes = [CLASSES[source] for source in sources]
    one_hot_encoder = utils.to_categorical(classes)
    return one_hot_encoder

#len of train data
N_train = int(dataset_size * 0.8)

#define train data
titles_train, sources_train = (titles_df.title[:N_train, ],
                               titles_df.source[:N_train, ])

#define validation data
titles_valid, sources_valid = (titles_df.title[N_train:, ],
                               titles_df.source[N_train:, ])

#count the values
sources_train.value_counts()
sources_valid.value_counts()

#preprocess the data
X_train, Y_train = pad_titles(titles_train), encode_labels(sources_train)
X_valid, Y_valid = pad_titles(titles_valid), encode_labels(sources_valid)

#print the first three data points
X_train[:3], Y_train[:3]

#Build a model DNN

def build_dnn(embed_dim):
    model = models.Sequential(
        [
            layers.Embedding(vocabulary_size + 1,
                             embed_dim,
                             # activity_regularizer=tf.keras.initializers.GlorotNormal(),
                             # embeddings_regularizer=tf.keras.regularizers.L1L2(l1=0.1, l2=0.1),
                             input_shape=[max_len]),
            layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
            layers.Dense(N_CLASSES, activation='softmax')
        ]
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

tf.random.set_seed(69)

MODEL_DIR = os.path.join(log_dir,'dnn')
BATCH_SIZE = 256
EPOCHS = 40
EMBED_DIM = 32
PATIENCE = 0

model = build_dnn(embed_dim=EMBED_DIM)

model_history = model.fit(X_train, Y_train, epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          validation_data=(X_valid, Y_valid),
                          callbacks=[callbacks.EarlyStopping(patience=PATIENCE),
                          callbacks.TensorBoard(MODEL_DIR)])


pd.DataFrame(model_history.history)[['loss', 'val_loss']].plot()
pd.DataFrame(model_history.history)[['accuracy', 'val_accuracy']].plot()

model.summary()
model.predict()
