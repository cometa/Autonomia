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

"""
  Simple steering and throttle prediction from a trained model and compared with a telemetry image.

  Usage:
    ./predict.py <DATA-DIR>

  Arguments:
    <DATA-DIR> image directory
"""

import cv2
import numpy as np
import sys
import os
import time
import math
from keras.models import model_from_json
from config import DataConfig
import utils
import ntpath

interactive =  False

# show an image in a proper scale
def show_img(img):
  screen_res = 320. * 2 , 240. * 2
  scale_width = screen_res[0] / img.shape[1]
  scale_height = screen_res[1] / img.shape[0]
  scale = min(scale_width, scale_height)
  window_width = int(img.shape[1] * scale)
  window_height = int(img.shape[0] * scale)
  cv2.namedWindow('dst1_rt', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('dst1_rt', window_width, window_height)
  cv2.imshow('dst1_rt', img)  
  return

if __name__ == "__main__":
  config = DataConfig()

  try:
    data_path = os.path.expanduser(sys.argv[1])
  except Exception, e:
    print e, "Usage: ./predict.py <DATA-DIR>"
    sys.exit(-1)

  if not os.path.exists(data_path):
    print "Directory %s not found." % data_path
    sys.exit(-1)

  # open labels csv file (frame filename, steering, throttle)
  with open("{}/labels.csv".format(data_path)) as f:
    labels = f.readlines()
  nlabels = len(labels)
  print "found %d labels" % nlabels

  out_file = open("{}/labels_pred.csv".format(data_path), 'w')

  # Load model structure

  model = model_from_json(open("{}/autonomia_cnn.json".format(data_path)).read())

  # Load model weights
  model.load_weights("{}/autonomia_cnn.h5".format(data_path))
  model.summary()

  img_height, img_width, num_channels = config.img_resample_dim[0], config.img_resample_dim[1], config.num_channels
  skip = config.skip_ahead

  for i,line in enumerate(labels):
    if i < skip:
      continue
    filename, steering, throttle= line.split(',')
    # image filename
    filename = data_path + '/' + filename
    # steering
    steering = int(steering)
    # throttle
    throttle = int(throttle)
    print filename, steering, throttle, utils.steering2bucket(steering), utils.throttle2bucket(throttle)
    # load image
    img = cv2.imread(filename)

    # convert to YCrCb
    gray_img =  cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)  

    if num_channels == 1:
      # extract and use Y plane only
      X_img, _, _ = cv2.split(gray_img)
    else:
      # use YCrCb
      X_img = gray_img

    if interactive: show_img(X_img)

    # crop image
    X_img = X_img[config.img_yaxis_start:config.img_yaxis_end + 1, config.img_xaxis_start:config.img_xaxis_end + 1]

    # resample image 
    X_img = cv2.resize(X_img, config.img_resample_dim, cv2.INTER_LINEAR)

    # X_img is of shape (1,:,:,:)
    X_img = X_img.reshape(1, img_height, img_width, num_channels)

    # normalize the image values
    X_img = X_img / 127.5 - 1

    now = time.time()
    # predict steering and throttle
    steering, throttle = model.predict(X_img[0:1])
    t = time.time() - now
    print "execution time:", t
#    steering = np.argmax(p[:, :15],  1)
#    throttle = np.argmax(p[:, 15:], 1)
#    print p[0, :15]
#    print p[0, 15:]

    steering = np.argmax(steering[0])
    throttle = np.argmax(throttle[0])

    print steering, throttle
    steering = utils.bucket2steering(steering)
    throttle = utils.bucket2throttle(throttle)
    print steering, throttle

    out_file.write("%s,%d,%d\n" % (ntpath.basename(filename), steering, throttle))

    if interactive: 
      key = cv2.waitKey(0)
      if key == 27:
          sys.exit(0)
