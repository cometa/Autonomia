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
from sklearn.utils import shuffle
from keras import callbacks
from keras.regularizers import l2

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

    return model

def create_model_2softmax(img_size):
    keep_rate = 0.3
    pool_size = (2, 2)
    img_input = Input(shape = img_size)
    x = Convolution2D(16, 5, 5, subsample=(2, 2), W_regularizer=l2(0.001), border_mode="same", activation='relu')(img_input)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Dropout(keep_rate)(x)
    x = Convolution2D(32, 2, 2, subsample=(1, 1), W_regularizer=l2(0.001), border_mode="valid", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    x = Flatten()(x)
    x = Dropout(keep_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(keep_rate)(x)

    o_st = Dense(num_outputs, activation='softmax', name='o_st')(x)
    o_thr = Dense(num_outputs, activation='softmax', name='o_thr')(x)
    model = Model(input=img_input, output=[o_st, o_thr])
    model.compile(optimizer='adam', loss={'o_st': 'categorical_crossentropy', 'o_thr': 'categorical_crossentropy'}, metrics=['accuracy'])

    return model

def create_modelB_2softmax(img_size):
    keep_rate = 0.5
    pool_size = (2, 2)
    img_input = Input(shape = img_size)
    x = Convolution2D(16, 5, 5, subsample=(1, 1), border_mode="same", activation='relu')(img_input)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Dropout(keep_rate)(x)
    x = Convolution2D(32, 5, 5, subsample=(1, 1), border_mode="same", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Dropout(keep_rate)(x)
    x = Convolution2D(64, 4, 4, subsample=(1, 1), border_mode="valid", activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Flatten()(x)
    x = Dropout(keep_rate)(x)

    x1 = Dense(900, activation='relu', W_regularizer=l2(0.001))(x)
    x1 = Dropout(keep_rate)(x1)
    x1 = Dense(110, activation='relu', W_regularizer=l2(0.001))(x1)
    x1 = Dropout(keep_rate)(x1)
    o_st = Dense(num_outputs, activation='softmax', name='o_st')(x1)

    x2 = Dense(800, activation='relu', W_regularizer=l2(0.001))(x)
    x2 = Dropout(keep_rate)(x2)
    x2 = Dense(128, activation='relu', W_regularizer=l2(0.001))(x2)
    x2 = Dropout(keep_rate)(x2)
    o_thr = Dense(num_outputs, activation='softmax', name='o_thr')(x2)
    model = Model(input=img_input, output=[o_st, o_thr])
    model.compile(optimizer='adam', loss={'o_st': 'categorical_crossentropy', 'o_thr': 'categorical_crossentropy'}, metrics=['accuracy'])

    return model

models = {
  'modelB_2softmax': create_modelB_2softmax,
  'model_2softmax': create_model_2softmax,
  'model_relu' : create_model_relu,
  'model_relu2' : create_model_relu2,  
}

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

  row, col = config.img_resample_dim
  ch = config.num_channels
  num_epoch = config.num_epoch
  batch_size = config.batch_size
  validation_split = config.validation_split
  num_outputs = config.num_buckets * 1

  # set of callbacks to save model weights during training when loss of validation set decreases
  #model_path = os.path.expanduser('model.h5')
  #Save the model after each epoch if the validation loss improved.
  save_best = callbacks.ModelCheckpoint("{}/autonomia_cnn_step.h5".format(data_path), monitor='val_loss', verbose=1, 
                                     save_best_only=True, mode='min')

  #stop training if the validation loss doesn't improve for 5 consecutive epochs.
  early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, 
                                     verbose=0, mode='auto')

  #callbacks_list = [save_best, early_stop]
  callbacks_list = []

  model = models[config.model]((row, col, ch))
  print("---------------------------")
  print("model %s is created and compiled\r\n" % config.model)
  print(model.summary())

  print("loading images and labels")
  X = np.load("{}/X_yuv_gray.npy".format(data_path))
  y1_steering = np.load("{}/y1_steering.npy".format(data_path))
  y2_throttle = np.load("{}/y2_throttle.npy".format(data_path))
  X, y1_steering, y2_throttle = shuffle(X, y1_steering, y2_throttle)
  # and trained it via:
  history = model.fit(X, {'o_st': y1_steering, 'o_thr': y2_throttle}, batch_size=batch_size, nb_epoch=num_epoch, verbose=1, validation_split=validation_split, callbacks=callbacks_list )
  
#  start_val = round(len(X)*0.8)
#  X_val = X[start_val:start_val + 200]
##  y_val = y1_steering[start_val:start_val + 200, :]
#  pred_val = np.array( model.predict(X_val, batch_size=batch_size) )
#  np.save('pred_validation.npy', np.hstack([y_val, pred_val[0,:,:]]))

  print("saving model and weights")
  with open("{}/autonomia_cnn.json".format(data_path), 'w') as f:
      f.write(model.to_json())

  model.save_weights("{}/autonomia_cnn.h5".format(data_path))
