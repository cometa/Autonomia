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
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Flatten

from config import TrainConfig
from keras import backend as K

def custom_objective(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true)

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

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model

if __name__ == "__main__":
    config = TrainConfig()

    ch = config.num_channels
    row = config.img_height
    col = config.img_width

    num_epoch = config.num_epoch
    batch_size = config.batch_size
    data_path = config.data_path

    num_outputs = config.num_buckets * 2

    model = create_model_prelu()
    print model.summary()
