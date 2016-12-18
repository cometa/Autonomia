#!/usr/bin/env python
"""
Extract regions of interest with digits from images.

An image has a 3 digit steering and 3 digit throttle ROI in the bottom left.
"""

import cv2
import numpy as np
import sys
import glob
import random

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
  return telemgray #thresh

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

def show_digits(digits):
  for i in range(0,6):
    b = np.array(digits[i]).reshape((7, 5, 1))
    window_width = int(b.shape[1] * 40.)
    window_height = int(b.shape[0] * 40.)
    #print "window size",window_width, window_height
    cv2.namedWindow('digit ' + str(i), cv2.WINDOW_NORMAL)
    cv2.resizeWindow('digit '+ str(i), window_width, window_height)
    cv2.imshow('digit '+ str(i), b)
  return

def main(argv):
  # scan directory 
  fnames = scan_dir(argv[0])
  print "found %d jpg files in %s" % (len(fnames), dir)

  # array of all images
  X = np.zeros(shape=(len(fnames) * 6, 7, 5), dtype=np.float)
  # array of ground thruth
  y_ = np.zeros(shape=(len(fnames) * 6, 10), dtype=np.float)

  row = 0
  num_images = 100
#  for filename in fnames:
  for i in range(1, num_images):
    filename = "%s/img%05d.jpg" % (argv[0],random.randrange(1, len(fnames)) )
    print filename
    img = cv2.imread(filename)
    show_img(img)
    roi = extract_roi(img)
    digits = get_digits(roi)

    for d in digits:
      bmap = np.array(d).reshape((7, 5, 1))
      cv2.namedWindow('digit', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('digit', 280, 200)
      cv2.imshow('digit', d)
      key = cv2.waitKey(0)
      if key == 27:
        return
      np.save('testset', X)
      np.save('testround', y_)

      num = int(key - 48)
      print num
      X[row] = d
      y_[row, num] = 1. 
      print X[row]
      print y_[row]
      row += 1



if __name__ == "__main__":
    main(sys.argv[1:])
