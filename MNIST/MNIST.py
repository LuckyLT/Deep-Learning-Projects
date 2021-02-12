import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.datasets import mnist

np.set_printoptions(suppress=True)

#Load the MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#EDA
print(X_train.shape) # 60000 28 28 --> not bad
print(X_train[0])

#print the first digit, looks like 5 :)
plt.imshow(X_train[0])
plt.title(y_train[0])

#Scale the data
X_train = X_train / 255.0
X_test = X_test / 255.0

#dtype of the x_train
print(X_train.dtype) #first uint8 --> float 64

#set input shape
i_shape = X_train.shape[1:]

#set Dropout rate
DROPOUT_RATE = 0.1

# define a initial Sequential model
model = Sequential([
                    Dense(256, input_shape=i_shape, activation="sigmoid"),
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(DROPOUT_RATE),
                    Dense(10, activation="softmax")
])

# define more sophisticated model
model_2 = Sequential([
                    Input(shape=i_shape),
                    Flatten(),
                    Dense(20, activation="relu"),
                    Dropout(DROPOUT_RATE),
                    Dense(50, activation="relu"),
                    Dropout(DROPOUT_RATE),
                    Dense(30, activation="relu"),
                    Dropout(DROPOUT_RATE),
                    Dense(10, activation="softmax")
])

#print the summaries
model.summary()
model_2.summary()

#compile the model with Adam optimizer and l_r = 0.001
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy()])

model_2.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy()])

#fit the model
history = model.fit(x=X_train, y=y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
history_2 = model_2.fit(x=X_train, y=y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

history_2.history

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model_2.evaluate(X_test, y_test, batch_size=32)
print("test loss, test acc:", results)


#plot loss and validation loss
plt.figure()
plt.plot(range(len(history_2.history["loss"])), history_2.history["loss"], c="g", label="train Loss")
plt.plot(range(len(history_2.history["val_loss"])), history_2.history["val_loss"], c="r", label="val Loss")
plt.xticks(list(range(len(history_2.history['loss']))))
plt.legend()
plt.title('Model 2 Training and Validation Loss')

#save the picture
plt.savefig('MNIST/Visualization/Model_2.png')

#save the model
model_2.save('MNIST/saved_model/model_2')
