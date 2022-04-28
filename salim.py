import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
plt.imshow(X_train[8], cmap = 'binary')

X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

import tensorflow as tf
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

salim_model = Sequential()
salim_model.add(Conv2D(32, (3,3), input_shape = (28,28,1), activation = 'relu'))
salim_model.add(MaxPool2D(2,2))


salim_model.add(Conv2D(64, (3,3), activation = 'relu'))
salim_model.add(MaxPool2D(2,2))

salim_model.add(Flatten())

salim_model.add(Dropout(0.25))

salim_model.add(Dense(10, activation = 'softmax'))
salim_model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor = 'val_acc', min_delta = 0.01, patience = 5, verbose = 1)

mc = ModelCheckpoint('./estmodel.h5', monitor = 'val_acc', verbose = 1, save_best_only = True) 

cb = [es,mc]
his = salim_model.fit(X_train, y_train, epochs = 5, validation_split = 0.25, callbacks = cb)
salim_model.save('my_model.h5')