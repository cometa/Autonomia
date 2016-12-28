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

from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K

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

  # Load model structure
  model = model_from_json(open("{}/autonomia_cnn.json".format(data_path)).read())

  # Load model weights
  model.load_weights("{}/autonomia_cnn_weights.h5".format(data_path))

  skip = 100
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
    print filename, steering, throttle
    # load image
    img = cv2.imread(filename)
    # convert to grayscale
    gray_img =  cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) 
    # extract Y plane
    Y_img, _, _ = cv2.split(gray_img)
    # Y_img is of shape (1,240,320,1)
    Y_img = Y_img.reshape(1, 240, 320, 1)
    # normalize the image values
    Y_img = Y_img / 255.

    # show image
    show_img(img)
    # predict steering and throttle
    p = model.predict(Y_img[0:1])
    print "p.shape", p.shape
    steering = np.argmax(p[:, :15],  1)
    throttle = np.argmax(p[:, 15:], 1)
    print p[0, :15]
    print p[0, 15:]
    steering = steering[0]
    throttle = throttle[0]
    print steering, throttle

    key = cv2.waitKey(0)
    if key == 27:
        sys.exit(0)