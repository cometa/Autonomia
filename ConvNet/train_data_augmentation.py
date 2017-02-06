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
import csv
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
import cv2
# utils.py is a link to ../utils.py
import utils
from sklearn.model_selection import train_test_split
SEED = 42

# scale image size
def img_array(img_adress, label):
    img = cv2.imread(img_adress)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return (img, label)


def image_hflip(img, label):
    '''
    Randomly flip image along horizontal axis: 1/2 chance that the image will be flipped
    img: original image in array type
    label: steering angle value of the original image
    '''
    choice = np.random.choice([0,1])
    if choice == 1:
            img = cv2.flip(img, 1)
            delta_label = label - 90
            label = 90 - delta_label 
    
    return (img, label)



def transformation_brightness(img):
    '''
    Adjust the brightness of the image, by a randomly generated factor between 0.1 (dark) and 1. (unchanged)
    img: original image in array type
    label: steering angle value of the original image
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #change Value/Brightness/Luminance: alpha * V
    alpha = np.random.uniform(low=0.5, high=1.0, size=None)
    v = hsv[:,:,2]
    v = v * alpha
    hsv[:,:,2] = v
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
    
    return rgb

            
def gamma_transform(img, label):
    '''
    Adjust the brightness of the image
    img: original image in array type
    label: steering angle value of the original image
    '''
    gamma = np.random.uniform(low=0.2, high=1.0, size=None)
    inv_gamma = 1.0/gamma
    img = 255 *(img/255.)**(inv_gamma)
    
    return (img, label)


def pixel_scaling(img, label):
    return ((img/127.5 - 1), label)


def image_transform(img_adress, label, target_sz ):  
    #open image as array from adress
    img, label = img_array(img_adress, label)
    # change luminance: 50% chance
    img, label = gamma_transform(img, label)
    # resize image
    # Here we take only Y component
    img = cv2.resize(img[:,:,0], target_sz, cv2.INTER_LINEAR)
    #Horizontal flip
    img, label = image_hflip(img, label)
    #pixel scaling
    img, label = pixel_scaling(img, label)
    return (img, label)



def batch_generator(x, y , num_buckets, dir_data, batch_size, img_sz, training=True, monitor=True, yieldXY=True):
    """
    Generate training batch: Yield X and Y data when the batch is filled.
    Data augmentation schemes: horizontal flip, chnage brightness, change viewpoint
    At each EPOCH, before data augmentation, del_rate=95% of examples with 
    steering angle = 0 are deleted from original dataset
    x: list of the adress of all images to be used for training
    y: steering angles
    training: use True for generating training batch with data augmentation scheme 
              use False to generate validation batch
    batch_size: size of the batch (X, Y)
    img_sz: size of the image (height, width, channel) to generate
    del_rate: percent of examples with steering angle=0 that will be deleted before generating batches
    monitor: save X of the last batch generated 'X_batch_sample.npy
             save angles of all batches generated 'y_bag.npy
    yieldXY: if True, generator yields (X, {Y1,Y2})
            otherwise, yields X only (useful for predict_generator()
    """

    if training:
        x, y = shuffle(x, y)
    offset = 0
    '''
    True as long as total number of examples generated is lower than the number of 'samples_per_epoch' set by user.
    '''
    while True: 
        # Initialize X and Y array
        X = np.zeros((batch_size, *img_sz), dtype=np.float64)
        Y_st = np.zeros((batch_size, num_buckets), dtype=np.float64)
        Y_th = np.zeros((batch_size, num_buckets), dtype=np.float64)
        # array of labels - steering and throttle as one-hot arrays
        #Generate a batch
        for example in range(batch_size):
            img_adress, label_st, label_th = dir_data +'/'+ x[example + offset], y[example + offset, 0], y[example + offset, 1]
            assert os.path.exists(img_adress), 'Image file ['+ img_adress +'] not found-'
            if training:
                #img, label = image_transformation(img_adress, label, (img_sz[0], img_sz[1]))
                img, label_st = image_transform(img_adress, label_st, target_sz=(img_sz[0], img_sz[1]) )
            else:
                img, label_st = image_transform(img_adress, label_st, target_sz=(img_sz[0], img_sz[1]) )

            # update batch X and Y array with new example
            X[example,:,:,0] = img
            Y_st[example, utils.steering2bucket(label_st)] = 1
            Y_th[example, utils.steering2bucket(label_th)] = 1
            # when reaching end of original dataset x, loop from start again
            # shuffle original dataset / randomly remove 95% of the example with steering angle = 0
            if (example + 1) + offset > len(y) - 1:
                x, y = shuffle(x, y)
                offset = 0
        if yieldXY:
            yield (X, {'o_st': Y_st, 'o_thr': Y_th})
        else:
            yield X
        
        offset = offset + batch_size
        

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
    # end of feature detector
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


  #############
  # Parameters
  ##########
  skip = config.skip_ahead
  train_img_sz = config.img_resample_dim
  train_img_ch = config.num_channels
  num_epoch = config.num_epoch
  batch_size = config.batch_size
  validation_split = config.validation_split
  num_outputs = config.num_buckets * 1
  data_augmentation = 100
  samples_per_epoch = batch_size * data_augmentation
  
  ##########
  # Data Preparation
  ###########
  print("loading train data csv")
  log_data = [] 
  with open(data_path+'/labels.csv', 'r') as csvfile:
    data_vid = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(data_vid):
        if i >= skip:
          log_data.append(row)

  # log = [img_adress, steering val, throttle val]
  log_data = np.array(log_data)
  x_ = log_data[:, 0] 
  y_ = log_data[:, 1::].astype(float)

  #clip y_steering: 60, 120 (check histogram in notebook)
  y_[:,0]=np.clip(y_[:,0], 60, 120)
  y_[:,1]=np.clip(y_[:,1], 60, 120)


  #shuffle data
  x_, y_= shuffle(x_, y_)
  # split train/validation set with ratio: 5:1
  X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=validation_split, random_state=SEED)

  print('Train set size: {} | Validation set size: {}'.format(len(X_train), len(X_val)))
  nb_val_samples = len(y_val) - len(y_val)%batch_size # make validation set size to be a multiple of batch_size 

  ####################
  # MODEL
  ######
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

  model = models[config.model]((*train_img_sz, train_img_ch))
  print("---------------------------")
  print("model %s is created and compiled\r\n" % config.model)
  print(model.summary())

  # and trained it via:
  #history = model.fit(X, , batch_size=batch_size, nb_epoch=num_epoch, verbose=1, validation_split=validation_split, callbacks=callbacks_list )
  history = model.fit_generator(batch_generator(X_train, y_train, num_buckets=num_outputs, dir_data=data_path, batch_size=batch_size, img_sz=(*train_img_sz, train_img_ch), training=True),
                              samples_per_epoch=samples_per_epoch, nb_val_samples=nb_val_samples,
                              validation_data=batch_generator(X_val, y_val, num_buckets=num_outputs, dir_data=data_path, batch_size=batch_size, img_sz=(*train_img_sz, train_img_ch), 
                                                              training=False, monitor=False),
                              nb_epoch=num_epoch, verbose=1, callbacks=callbacks_list)


  print("saving model and weights")
  with open("{}/autonomia_cnn.json".format(data_path), 'w') as f:
      f.write(model.to_json())

  model.save_weights("{}/autonomia_cnn.h5".format(data_path))

  #clear session to avoid error at the end of program: "AttributeError: 'NoneType' object has no attribute 'TF_DeleteStatus'"
    # The alternative does not work: import gc; gc.collect()
    # https://github.com/tensorflow/tensorflow/issues/3388
  K.clear_session()
