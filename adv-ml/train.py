from keras.models import Sequential
from tensorflow.keras.models import  load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
import numpy as np
SEED =42
def create_model(p_activation="linear"):
  model = Sequential()

  model.add(Conv2D(32, (3, 3),activation="relu", input_shape=(28, 28, 1)))

  model.add(Conv2D(32, (3, 3), activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, (3, 3), activation="relu"))
  model.add(Conv2D(64, (3, 3), activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(200, activation="relu"))
  model.add(Dense(200, activation="relu"))
  model.add(Dense(10, activation=p_activation))

  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss =
                tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=sgd, metrics=["accuracy"])
  return model
