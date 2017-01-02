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
  Extract steering and throttle values embedded in an image.

  Usage:
    ./extract_telemetry <IMAGE-DIR>

  Arguments:
    <IMAGE-DIR> image directory

  Output: 
    labels.csv. A CSV file in <IMAGE-DIR> with filename, steering and throttle extracted and interpreted.
"""

import cv2
import numpy as np
import sys
import glob
import random
import time
import ntpath

from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.utils import np_utils
from keras.models import model_from_json

interactive = False

def scan_dir(dir):
  """ Scan *.jpg files from directory dir """
  pattern = dir + '/*.jpg'
  fnames = glob.glob(pattern)
  return fnames

def extract_roi(img):
  height, width = img.shape[:2]
  # telemetry is embedded in a 40x10 rectangle in the bottom left corner
  telem_w = 40
  telem_h = 10
  # extract the rectangle with steering and throttle into a separate image
  telem = img[height - telem_h - 1:height - 1, 0:telem_w - 1]
  # convert to grayscale
  telemgray = cv2.cvtColor(telem, cv2.COLOR_BGR2GRAY)
  # threshold the image
  _, thresh = cv2.threshold(telemgray,120,255,0)  
  return thresh #telemgray 

def get_digits(img):
  a = np.zeros(6, dtype=object)
  # extract six digits from the telem image
  y = 2
  w = 5
  h = 7
  for i in range(0, 6):
    x = i * 6
    if i >= 3:
      x += 4
    else:
      x += 1
    # the digit bounding box position and dimenstions are x,y,w,h
    # extract a letter from the image
    a[i] = img[y:y+h, x:x+w]
  return a

def show_img(img):
  screen_res = 320. * 2 , 240. * 2
  scale_width = screen_res[0] / img.shape[1]
  scale_height = screen_res[1] / img.shape[0]
  scale = min(scale_width, scale_height)
  window_width = int(img.shape[1] * scale)
  window_height = int(img.shape[0] * scale)
  cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('dst_rt', window_width, window_height)
  cv2.imshow('dst_rt', img)  
  return

def show_roi(img):
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


def main(argv):
  # scan directory 
  f = scan_dir(argv[0])
  fnames = sorted(f)
  print "found %d jpg files in %s" % (len(fnames), dir)

  out_file = open(argv[0] + '/labels.csv', 'w')

  # Load model structure
  model = model_from_json(open('DataPrep/DigitsNet/digits_cnn.json').read())

  # Load model weights
  model.load_weights('DataPrep/DigitsNet/digits_cnn_weights.h5')

  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

  digits = np.zeros(6, dtype=int)
  for filename in fnames:
    img = cv2.imread(filename)
    if interactive: show_img(img)
    roi = extract_roi(img)
    if interactive: show_roi(roi)

    digit_imgs = get_digits(roi)
    if len(digit_imgs) != 6:
      print "ERROR: less than six digits in the image"
      key = cv2.waitKey(0)
      continue

    for i, digim in enumerate(digit_imgs):
      X = np.array(digim,  dtype=np.float)
      X /= 255.
      X = digim.reshape(1,7,5,1)

      p = model.predict(X[0:1])
      digits[i] = np.argmax(p, axis=1)[0]

    steering = digits[0] * 10**2 + digits[1] * 10 + digits[2]
    throttle = digits[3] * 10**2 + digits[4] * 10 + digits[5]

    print filename, steering, throttle
    out_file.write("%s,%d,%d\n" % (ntpath.basename(filename), steering, throttle))

    if interactive:
      key = cv2.waitKey(0)
      if key == 27:
        return

  out_file.close()
if __name__ == "__main__":
    main(sys.argv[1:])
