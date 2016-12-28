"""
  Create and train a CNN to predict digits. The input files contain a numpy array of images and labels.
  The output files contain the model and the weigths: digits_cnn.json, digits_cnn.h5.

  $ python train_digits_cnn {DATASET} {LABELS}

  Input files: dataset-1184.npy, labels-1184.npy
  
"""

import numpy as np
import sys
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Flatten

def main (argv):
  X=np.load(argv[0]) # 'dataset-1184.npy'
  Y=np.load(argv[1]) # 'labels-1184.npy'

  train_index = X.shape[0] * 0.8
  X_train = X[0:train_index]
  X_test = X[train_index + 1:X.shape[0] - 1]

  y_train = Y[0:train_index]
  y_test = Y[train_index + 1:Y.shape[0] - 1]

  #------------

  # reshaping and normalizing the images
  X_train = X_train.reshape(X_train.shape[0], 7, 5, 1)
  X_test = X_test.reshape(X_test.shape[0], 7, 5, 1)
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255.
  X_test /= 255.

  #------------

  model = Sequential()
  # number of convolutional filters to use
  nb_filters = 32
  # convolution kernel size
  kernel_size = (2, 2)
  # size of pooling area for max pooling
  pool_size = (3, 3)

  batch_size = 64
  nb_classes = 10
  nb_epoch = 30

  # Design a CNN
  model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                          border_mode='same', input_shape=(7, 5, 1)))
  model.add(Activation('tanh'))

  model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
  model.add(Activation('relu'))

  model.add(MaxPooling2D(pool_size=pool_size))
  #model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(128, init='he_normal'))
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  model.add(Dense(nb_classes, init='he_normal'))
  model.add(Activation('softmax'))

  sgd = RMSprop(lr=0.001)

  from keras import backend as K
  def custom_objective(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true)

#  model.compile(loss='categorical_crossentropy',
  model.compile(loss=custom_objective,
                optimizer=sgd, #'rmsprop',
                metrics=['accuracy'])

  model.summary()

  #------------

  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2)

  score = model.evaluate(X_test, y_test, verbose=2)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])

  with open('digits_cnn.json', 'w') as f:
      f.write(model.to_json())

  model.save_weights('digits_cnn_weights.h5')

if __name__ == "__main__":
    main(sys.argv[1:])
