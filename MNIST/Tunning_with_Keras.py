import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

#load the data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

#Scale the data between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

# Instantiate the tuner and perform hypertuning
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3, #Reduction factor for the number of epochs and number of models for each bracket.
                     directory='MNIST//hypertunning',
                     project_name='intro_to_kt')

# Create an early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
#patience is Number of epochs with no improvement after which training will be stopped.

#search through all the hyper parameters
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

#best hyper parameters are saved in best hps
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Train the model with the best hyper parameters, seeking the optimal number of epochs
model = tuner.hypermodel.build(best_hps)
model.summary()

#safe the history for 50 epochs
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

#find the max validation acc per epoch
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print(f'Best epoch: {best_epoch}')

# Re-instantiate the hypermodel and train it with the optimal number of epochs from above
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(X_train, y_train, epochs=best_epoch, callbacks=[stop_early])

#evaluate the model
eval_result = hypermodel.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)
# [test loss, test accuracy]: [0.10527059435844421, 0.9800999760627747]

#save the model in folder below
hypermodel.save('MNIST/models/best_model')