#!/usr/bin/env python
"""
  Cloud connected autonomous RC car.

  Copyright 2016 Visible Energy Inc. All Rights Reserved.
"""
__license__ = """
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import sys
import os
#from keras.models import Sequential, Graph
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Convolution2D, MaxPooling2D, AveragePooling2D, Flatten, PReLU
from keras.models import Sequential, Model
from config import TrainConfig
from keras import backend as K

def combined_crossentropy(y_true, y_pred):
    y_true_steering = y_true[:, :num_outputs]
    y_true_throttle = y_true[:, num_outputs:]
    y_pred_steering = y_pred[:, :num_outputs]
    y_pred_throttle = y_pred[:, num_outputs:]

    steering_crossentropy = K.categorical_crossentropy(y_pred_steering, y_true_steering)
    throttle_crossentropy = K.categorical_crossentropy(y_pred_throttle, y_true_throttle)
    return (steering_crossentropy + throttle_crossentropy) / 2.






def create_model_relu2():
    # size of pooling area for max pooling
    pool_size = (2, 2)

    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(row, col, ch)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(256, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Dense(num_outputs, init='he_normal'))
    model.add(Activation('softmax'))

    sgd = RMSprop(lr=0.001)
    model.compile(optimizer=sgd, loss=combined_crossentropy, metrics=['accuracy'])

    print('Model relu2 is created and compiled..')
    return model

def create_model_relu():
    pool_size = (2, 2)
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(row, col, ch)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size)) #added

    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Activation('relu'))
#    model.add(Dense(512, init='he_normal'))
    model.add(Dense(256, init='he_normal')) #mod
    model.add(Dropout(.5))
    model.add(Activation('relu'))

    model.add(Dense(num_outputs, init='he_normal'))
    model.add(Activation('softmax'))

    sgd = RMSprop(lr=0.001)
#    sgd = Adam(lr=0.001)  #mod
    model.compile(optimizer=sgd, loss=combined_crossentropy, metrics=['accuracy'])

    print('Model relu is created and compiled..')
    return model

def create_model_prelu():
    model = Sequential()

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(row, col, ch)))
    model.add(PReLU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(PReLU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    #model.add(Dropout(.5))
    model.add(PReLU())
    model.add(Dense(512, init='he_normal'))
    #model.add(Dropout(.5))
    model.add(PReLU())

    model.add(Dense(num_outputs, init='he_normal'))
    model.add(Activation('softmax'))

    sgd = RMSprop(lr=0.001)
    model.compile(optimizer=sgd, loss=combined_crossentropy, metrics=['accuracy'])

    print('Model prelu is created and compiled..')
    return model


def create_model_2softmax(img_size):
    
    pool_size = (2, 2)
    print('Number of outputs:', num_outputs)
    img_input = Input(shape=(150, 320, 1))
    x = Convolution2D(16, 5, 5, subsample=(2, 2), border_mode="same", activation='relu')(img_input)
    x = MaxPooling2D(pool_size=pool_size)(x)
    #x = Dropout(0.5)(x)
    x = Convolution2D(32, 2, 2, subsample=(1, 1), border_mode="same", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    #x = Dropout(0.33)(x)
    o_st = Dense(num_outputs, activation='softmax', name='o_st')(x)
    o_thr = Dense(num_outputs, activation='softmax', name='o_thr')(x)
    model = Model(input=img_input, output=[o_st, o_thr])
    model.compile(optimizer='adam', loss={'o_st': 'categorical_crossentropy', 'o_thr': 'categorical_crossentropy'}, metrics=['accuracy'])

    return model

if __name__ == "__main__":
  config = TrainConfig()

  try:
    data_path = os.path.expanduser(sys.argv[1])
  except Exception as e:
    print(e, "Usage: ./prepare_data.py <DATA-DIR>")
    sys.exit(-1)

  if not os.path.exists(data_path):
    print("Directory %s not found." % data_path)
    sys.exit(-1)

  row, col = config.img_height, config.img_width
  ch = config.num_channels
  num_epoch = config.num_epoch
  batch_size = config.batch_size
  num_outputs = config.num_buckets * 1

  model = create_model_2softmax( (row, col, ch) )
  print(model.summary())

  print("loading images and labels")
  X = np.load("{}/X_yuv_gray.npy".format(data_path))-0.5
  y1_steering = np.load("{}/y1_steering.npy".format(data_path))
  y2_throttle = np.load("{}/y2_throttle.npy".format(data_path))
  # and trained it via:
  history = model.fit(X, {'o_st': y1_steering, 'o_thr': y2_throttle}, batch_size=batch_size, nb_epoch=30, verbose=1, validation_split=0.30 )

  print("saving model and weights")
  with open("{}/autonomia_cnn.json".format(data_path), 'w') as f:
      f.write(model.to_json())

  model.save_weights("{}/autonomia_cnn.h5".format(data_path))
