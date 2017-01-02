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
#from keras.models import Sequential
from keras.models import model_from_json

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

# throttle bucket conversion map -- from [0,180] range to a bucket number in the [0.14] range
throttle_map = [
    [80,0], # if t <= 80 -> o=0 # Breaking:
    [82,1], # elif t <= 82 -> o=1
    [84,2], # elif t <= 84 -> o=2
    [86,3], # elif t <= 86 -> o=3
    [87,4], # elif t <= 87 -> o=4 # Breaking ^
    
    [96,5], # elif t <= 96 -> o=5 # Neutral

    [97,6], # elif t <= 97 -> o=6 # Forward:
    [98,7], # elif t <= 98 -> o=7
    [99,8], # elif t <= 99 -> o=8
    [100,9], # elif t <= 100 -> o=9
    
    [101,10], # elif t <= 101 -> o=10
    [102,11], # elif t <= 102 -> o=11
    [105,12], # elif t <= 105 -> o=12
    [107,13], # elif t <= 107 -> o=13
    [110,14]  # elif t <= 110 -> o=14
]

def bucket2throttle(t):
    """ Reverse the function that buckets the throttle for neural net output """
    map_back = {5:90}
    t = int(float(t)+0.5)
    for ibucket,(max_in_bucket,bucket) in enumerate(throttle_map):
        if t == bucket:
            if map_back.has_key(bucket):
                return map_back[bucket]

            return max_in_bucket
    return 100 # Never happens, defensively select a mild acceleration

def bucket2steering(a):
    """ Reverse the function that buckets the steering for neural net output """
    steer = a - 7
    original = steer
    steer = abs(steer)
    steer = math.pow(2.0, steer)
    steer -= 1.0
    steer = math.copysign(steer, original)
    steer += 90.0
    steer = max(0, min(179, steer))
    return steer

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
  model.load_weights("{}/autonomia_cnn.h5".format(data_path))
  model.summary()

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

    Y_img = Y_img[80:230,0:320]
    # Y_img is of shape (1,240,320,1)
#    Y_img = Y_img.reshape(1, 240, 320, 1)
    Y_img = Y_img.reshape(1, 150, 320, 1)
    
    # normalize the image values
    Y_img = Y_img / 255.

    # show image
    show_img(img)
    now = time.time()
    # predict steering and throttle
    p = model.predict(Y_img[0:1])
    t = time.time() - now
    print "execution time:", t
    steering = np.argmax(p[:, :15],  1)
    throttle = np.argmax(p[:, 15:], 1)
    print p[0, :15]
    print p[0, 15:]
    steering = steering[0]
    throttle = throttle[0]

    print steering, throttle
    steering = bucket2steering(steering)
    throttle = bucket2throttle(throttle)
    print steering, throttle

    key = cv2.waitKey(0)

    if key == 27:
        sys.exit(0)