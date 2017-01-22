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

# utils.py is a link to ../utils.py
import utils

interactive = False # True

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
  except Exception as e:
    print("Usage: ./prepare_data.py <DATA-DIR>")
    sys.exit(-1)

  if not os.path.exists(data_path):
    print("Directory %s not found." % data_path)
    sys.exit(-1)

  row, col = config.img_height, config.img_width
  num_buckets = config.num_buckets
  img_height, img_width, num_channels = config.img_resample_dim[0], config.img_resample_dim[1], config.num_channels

  # open labels csv file (frame filename, steering, throttle)
  with open("{}/labels.csv".format(data_path)) as f:
    labels = f.readlines()
  nlabels = len(labels)
  print("-----------------------------------")
  print("found %d labels" % nlabels)
  print("image height: %d width: %d" % (config.img_height, config.img_width))
  print("-----------------------------------\r\n")

  # telemetry is ahead of `config.skip` number of frames
  skip = config.skip_ahead

  # array of all images - width, height 
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
    # convert to YCrCb
    gray_img =  cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)  

    if num_channels == 1:
      # extract and use Y plane only
      X_img, _, _ = cv2.split(gray_img)
  #    Y_img = Y_img[80:230,0:320]
  #    Y_img = Y_img[140:230,0:320]
    else:
      # use YCrCb
      X_img = gray_img

    # crop image
    X_img = X_img[config.img_yaxis_start:config.img_yaxis_end + 1, config.img_xaxis_start:config.img_xaxis_end + 1]

    # resample image 
    X_img = cv2.resize(X_img, config.img_resample_dim, cv2.INTER_LINEAR)

    if interactive: show_img(X_img)
    print(index)

    # X_img is of shape (1,:,:,:)
    X_img = X_img.reshape(1, img_height, img_width, num_channels)

    # normalize the image values
    X[index] = X_img / 127.5 - 1

    # steering bucket
    y1[index, utils.steering2bucket(steering)] = 1.
    # throttle bucket
    y2[index, utils.throttle2bucket(throttle)] = 1.

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
