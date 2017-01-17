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
  Prepare data for CNN training from image and label input files as specified in the labels.csv file obtained with the `extract_telemetry.py` utility.
  The `labels.csv`and the image files are in the DATA-DIR passed as first parameter. asconfig.data_path directory set in config.py. 

  This utility prepares three numpy arrays that are saved in the `X_yuv_gray.npy`, `y1_steering.npy` `y2_throttle.npy` files in the same <DATA-DIR> directory.
  The files contains respectively the processed images, the steering and throttle values for each image, encoded as one-hot vectors.
 
  Usage: 
    ./prepare_data.py <DATA-DIR> 

  Arguments:
    <DATA-DIR> input and output directory
"""

import cv2
import numpy as np
import sys
import os
import glob
import random
import time
import ntpath
import math
from config import DataConfig

interactive = False #True

# show an image in a proper scale
def show_gray(img):
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

# convert from [-90,90] range to a bucket number in the [0,14] range with log distribution to stretch the range of the buckets around 0.
def steering2bucket(s):
    s -= 90
    return int(round(math.copysign(math.log(abs(s) + 1, 2.0), s))) + 7

# convert throttle from [0,180] range to a bucket number in the [0.14] range using a map
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

def throttle2bucket(t):
  for max_in_bucket,bucket in throttle_map:
      if t <= max_in_bucket:
          return bucket
  return 14

if __name__ == "__main__":
  config = DataConfig()

  try:
    data_path = os.path.expanduser(sys.argv[1])
  except Exception as e:
    print("Usage: ./prepare_data.py <DATA-DIR>")
    sys.exit(-1)

  if not os.path.exists(data_path):
    print("Directory %s not found." % data_path)
    sys.exit(-1)

  row, col = config.img_height, config.img_width
  num_buckets = config.num_buckets
  img_height, img_width, num_channels = config.img_height, config.img_width, config.num_channels

  # open labels csv file (frame filename, steering, throttle)
  with open("{}/labels.csv".format(data_path)) as f:
    labels = f.readlines()
  nlabels = len(labels)
  print("found %d labels" % nlabels)

  # telemetry is ahead of `skip` number of frames
  skip = 1 #1

  # array of all images - width 240, height 
  X = np.zeros(shape=(nlabels - skip, img_height, img_width, num_channels), dtype=np.float64)

  # array of labels - steering and throttle as one-hot arrays
  y1 = np.zeros(shape=(nlabels - skip, num_buckets), dtype=np.float64)
  y2 = np.zeros(shape=(nlabels - skip, num_buckets), dtype=np.float64)

  for i,line in enumerate(labels):
    if i < skip:
      continue
    filename, _, _ = line.split(',')
    # image filename
    filename = data_path + '/' + filename
    index = i - skip
    # telemetery image
    label_telemetry = labels[index]
    _, steering, throttle = label_telemetry.split(',') 
    # steering
    steering = int(steering)
    # throttle
    throttle = int(throttle)
    print(filename, steering, throttle)

    img = cv2.imread(filename)
    # convert to grayscale
    gray_img =  cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) 
    # extract Y plane
    Y_img, _, _ = cv2.split(gray_img)
    Y_img = Y_img[80:230,0:320]

    if interactive: show_gray(Y_img)
    print(index)
    # Y_img is of shape (240,320,1)
    Y_img = Y_img.reshape(1, img_height, img_width, num_channels)
    # normalize the image values
    X[index] = Y_img / 255.
    # steering bucket
    y1[index, steering2bucket(steering)] = 1.
    # throttle bucket
    y2[index, throttle2bucket(throttle)] = 1.

    if interactive:
      key = cv2.waitKey(0)
      if key == 27:
        sys.exit(0)

  # =save arrays
  outpath = os.path.expanduser(sys.argv[1])
  if not os.path.exists(outpath):
      os.makedirs(outpath)

  print("saving images numpy array in: %s/X_yuv_gray.npy" % outpath )
  np.save("{}/X_yuv_gray".format(outpath), X)
  print("saving steering numpy array in: %s/y1_steering.npy" % outpath )
  np.save("{}/y1_steering".format(outpath), y1)
  print("saving throttle numpy array in: %s/y2_throttle.npy" % outpath )
  np.save("{}/y2_throttle".format(outpath), y2)
