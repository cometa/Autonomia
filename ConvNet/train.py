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
import cv2
import glob
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
import cnnModels as nnet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DriveLog():

  def __init__(self, folder_imgs):
    self.folder_imgs = folder_imgs

  def exists(self, fname='log.npy'):
      file_ls = glob.glob('*.npy')
      if fname in file_ls:
        return True
      else:
        print('Unable to find the log file: {}'.format(fname))
        return False



  def make_imgName(self, id):
    '''
    generate image filename from index in 1st column of summary txt file
    '''
    idx = '{:05d}'.format(id)
    img_name = 'img' + str(idx)  + '.jpg'
    return img_name


  def make_log(self):
    '''
    row {"c":1,"s":94,"time":"1489255712171","t":99,"device_id":"B827EB4879D6"}
    '''
    #get summary txt file
    summary_fileList = glob.glob(self.folder_imgs+'/*.txt')
    if len(summary_fileList) == 1:
      summary_txt = summary_fileList[0]
      print(summary_txt)
      with open(summary_txt, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        log = []
        for row in data:
            img_idx, steering, throttle = row[0][5::], row[1][2::], row[3][2::]
            img_fname = self.folder_imgs +'/'+ self.make_imgName( int(img_idx) )
            log.append([img_fname, steering, throttle])
        np.save('log.npy', log)
        return log
    else:
      print('Unable to find the summary file {}'.format('*'+self.folder_imgs+'*.txt'))


def steering2bucket(s, bucket_sz):
  """ 
  Convert from [0,180] range to a bucket number in the [0,14] 
  range with log distribution to stretch the range of the buckets around 0 
  """
  s -= 90
  return int(round(math.copysign(math.log(abs(s) + 1, 2.0), s))) + bucket_sz/2

def bucket2steering(a, bucket_sz):
  """ Reverse the function that buckets the steering for neural net output """
  steer = a - bucket_sz/2
  original = steer
  steer = abs(steer)
  steer = math.pow(2.0, steer)
  steer -= 1.0
  steer = math.copysign(steer, original)
  steer += 90.0
  steer = max(0, min(179, steer))
  return steer


def image_flip(img, steering):
  coin = np.random.choice([0, 1])
  if coin == 1:
     new_img = cv2.flip(img, 1)
     new_steering = - steering
     return (new_img, new_steering)
  else: 
     return (img, steering)


def batch_generator(x, y, batch_size, model_img_sz, n_outputs, ycrop_range = [120, -20], cspace='YCR_CB', model_type='classification', run='train'):
  '''
  Generate training batch: Yield X and Y data when the batch is filled.
  x: list of the adress of all images to be used for training
  y: steering angles
  batch_size: size of the batch (X, Y)
  model_img_sz: size of the image (height, width, channel) to generate
  monitor: save X of the last batch generated 'X_batch_sample.npy
      save angles of all batches generated 'y_bag.npy
  True as long as total number of examples generated is lower than the number of 'samples_per_epoch' set by user.
  '''
  offset = 0
  while True:
    # Initialize X and Y array
    X = np.zeros((batch_size, model_img_sz[0], model_img_sz[1]), dtype='float32')
    Y = np.zeros((batch_size, n_outputs), dtype='float32')
    #Generate a batch
    for example in range(batch_size):
      fname = x[example + offset]
      img = cv2.imread(fname)
      img = cv2.cvtColor(img, eval('cv2.COLOR_BGR2'+cspace) )
      if model_img_sz[2] == 1:
        img = img[:,:,0]
      #cv2 resize (x_size, y_size)
      img_resize = cv2.resize(img[ycrop_range[0]:ycrop_range[1], :], 
                      (model_img_sz[1], model_img_sz[0]), cv2.INTER_LINEAR)
      if run=='train':
        img_feed, steering = image_flip(img_resize, y[example+offset])
      else:
        img_feed, steering = img_resize, y[example + offset]
      if model_img_sz[2] == 1:
        X[example,:,:,0] = img_feed/255.0 - 0.5
      else:
        X[example,:,:,:] = img_feed/255.0 - 0.5
      if model_type == 'classification':
        steering_class = steering2bucket( steering, n_outputs )
        Y[example, int(steering_class)] = 1
      else:
        Y[example] = steering
        
    yield (X, Y)

    if (offset+batch_size >= len(y)-len(y)%batch_size): 
      offset = 0
      x, y = shuffle(x, y)
    else: 
      offset = offset + batch_size
    np.save('x_val.npy', X )
    np.save('y_val.npy', Y) #save last batch of images




models = {
  'model_wroscoe_mod': nnet.model_wroscoe_mod,
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

  ################
  # PARAMETERS
  img_resample_dim = config.img_resample_dim
  ch = config.num_channels
  num_epoch = config.num_epoch
  batch_size = config.batch_size
  validation_split = config.validation_split
  num_outputs = config.num_buckets * 1
  model_type = config.model_type
  data_augmentation = config.data_augmentation
  ycrop_range = config.ycrop_range
  cspace = config.cspace
  SEED = config.seed

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

  model = models[config.model]((img_resample_dim[0],img_resample_dim[1],ch), config.num_buckets, keep_rate=config.keep_rate, reg_fc=config.reg_fc, reg_conv=config.reg_conv, model_type=config.model_type)
  print("---------------------------")
  print("model %s is created and compiled\r\n" % config.model)
  print(model.summary())


  ########
  # Generate log array
  print("Create log book")
  logBook = DriveLog(data_path)

  #if logBook.exists():
  #  log = np.load('log.npy')
  #else:
  log = np.array( logBook.make_log() )
  print(log)
  x_original =  log[:, 0]
  yst_original = (log[:, 1].astype('float32'))
  # y is corrected steering angle
  x, yst = shuffle(x_original, yst_original-90)
  # split train/validation set with ratio: 5:1
  X_train, X_val, yst_train, yst_val = train_test_split(x, yst, test_size=validation_split, random_state=SEED)
  print('--> Train set size: {} | Validation set size: {}'.format(len(X_train), len(X_val)))
  
  #samples_per_epoch = len(y_train) * data_augmentation
  samples_per_epoch = ((len(yst_train) - len(yst_train)%batch_size)*data_augmentation)/128
  # make validation set size to be a multiple of batch_size
  nb_val_samples = len(yst_val) - len(yst_val)%batch_size
  nb_val_samples = 256
  # and trained it via:
  #history = model.fit(X, {'o_st': y1_steering, 'o_thr': y2_throttle}, batch_size=batch_size, nb_epoch=num_epoch, verbose=1, validation_split=validation_split, callbacks=callbacks_list )
  #history = model.fit_generator(batch_generator(X_train, yst_train, batch_size=batch_size, 
  #                      model_img_sz=(*img_resample_dim,ch), n_outputs=num_outputs, 
  #                     ycrop_range= ycrop_range, cspace=cspace, model_type=model_type, run='train'),
  #              steps_per_epoch=int(samples_per_epoch), nb_val_samples=nb_val_samples,
  #             validation_data=batch_generator(X_val, yst_val, batch_size=batch_size,     
  #                             model_img_sz = (*img_resample_dim,ch),
  #                             n_outputs=num_outputs, ycrop_range= ycrop_range, cspace=cspace, model_type=model_type, run='valid'),
  #              nb_epoch=num_epoch, verbose=1, callbacks=callbacks_list)
  
  history = model.fit_generator(batch_generator(X_train, yst_train, batch_size=batch_size, 
                        model_img_sz=(img_resample_dim[0], img_resample_dim[1], ch), n_outputs=num_outputs, 
                       ycrop_range= ycrop_range, cspace=cspace, model_type=model_type, run='train'),
                steps_per_epoch=int(samples_per_epoch),
                nb_epoch=num_epoch, verbose=1, callbacks=callbacks_list)

  print("saving model and weights")
  with open("{}/autonomia_cnn.json".format(data_path), 'w') as f:
      f.write(model.to_json())

  model.save_weights("{}/autonomia_cnn.h5".format(data_path))

  test_sz = 600
  start = 656
  x_test = x_original[start:start + test_sz]
  print(x_original.shape)
  y_test = yst_original[start:start + test_sz]-90
  X_test = np.zeros((test_sz, img_resample_dim[0], img_resample_dim[1], ch), dtype='float32')
  Y_test = np.zeros((test_sz, num_outputs), dtype='float32')
  for index, fname in enumerate(x_test):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, eval('cv2.COLOR_BGR2'+cspace) )
    if ch == 1:
      img = img[:,:,0]
    img = img[ycrop_range[0]:ycrop_range[1], :]
    img_resize = cv2.resize(img, (img_resample_dim[1], img_resample_dim[0]) )
    if model_type == 'classification':
      Y_test[index] = steering2bucket( y_test[index], n_buckets )
    else:
      Y_test[index] = y_test[index]

    if ch == 1:
      X_test[index, :,:,0] = img_resize/255.0-0.5

    else:
      X_test[index, :,:,:] = img_resize/255.0-0.5 
  pred = model.predict(X_test, batch_size=test_sz, verbose=1)
  if model_type == 'classification': 
    pred_class = np.argmax(pred, axis=1)
  else:
    pred_class = pred
  plt.plot(np.arange(0,test_sz, 1), Y_test, 'b-')
  print(pred_class)
  plt.plot(np.arange(0,test_sz, 1), pred_class, 'r-')
  plt.savefig('test_on_trainingset.png')
  plt.show()

  logfile_test = 'log_test.npy'
  test_dir = 'data/oakland170418'
  summary_file = 'data/oakland170418/1492543329.txt'
  files_log = glob.glob('*.npy')
  if logfile_test in files_log:
    try:
      log_test = np.load(logfile_test)
    except:
      print('file {} not found'.format(logfile_test))
  else:
    logBookTest = DriveLog(test_dir)
    log_test = np.array( logBookTest.make_log() )
  
  
  xt_original =  log_test[:, 0]
  yt_original = (log_test[:, 1].astype('float32')-90)
  start2 = 200
  x_t = xt_original[start2:start2 + test_sz]
  y_t = yt_original[start2:start2 + test_sz]
  X_t = np.zeros((test_sz, img_resample_dim[0],img_resample_dim[1], ch), dtype='float32')
  Y_t = np.zeros((test_sz, num_outputs), dtype='float32')
  for index, fname in enumerate(x_t):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, eval('cv2.COLOR_BGR2'+cspace) )
    if ch == 1:
      img = img[:,:,0]
    img = img[ycrop_range[0]:ycrop_range[1], :]
    img_resize = cv2.resize(img, img_resample_dim[::-1] )  #reverse tuple dimension
    if model_type == 'classification':
      Y_t[index] = steering2bucket( y_t[index], n_buckets )
    else:
      Y_t[index] = y_t[index]

    if ch == 1:
      X_t[index, :,:,0] = img_resize/255.0-0.5 
    else:
      X_t[index, :,:,:] = img_resize/255.0-0.5 
  pred = model.predict(X_t, batch_size=test_sz, verbose=1)
  if model_type == 'classification': 
    pred_class = np.argmax(pred, axis=1)
  else:
    pred_class = pred
  plt.plot(np.arange(0,test_sz, 1), Y_t, 'b-')
  plt.plot(np.arange(0,test_sz, 1), pred_class, 'r-')
  plt.savefig('test_on_testset.png')
  plt.show()


 

#clear session to avoid error at the end of program: "AttributeError: 'NoneType' object has no attribute 'TF_DeleteStatus'"
    # The alternative does not work: import gc; gc.collect()
    # https://github.com/tensorflow/tensorflow/issues/3388
  K.clear_session()