#!/usr/bin/env python

import math
import hashlib
import numpy as np
import cv2
from config import DataConfig

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

# images are aquired by ffmpeg -s 320x240 -pix_fmt  yuyv422 
def read_uyvy(filename, config, rows=240, cols=320):
    # input image size 
    image_size = rows * cols * 2
    # read a YUYV raw image and extract the Y plane - YUV 4:2:2 - (Y0,U0,Y1,V0),(Y2,U2,Y3,V2)
    # this is equivalent to YUY2 pixel format http://www.fourcc.org/pixel-format/yuv-yuy2
    fd = open(filename,'rb')
    f = np.fromfile(fd, dtype=np.uint8, count=image_size)
    print "read %d bytes from %s" % (len(f), filename)
    if len(f) != image_size: #rows*cols*2
        print "error in reading"
        return None

    # TODO: support for three channels YUV
    f = f.reshape((rows * cols / 2), 4)
    Y = np.empty((rows * cols), dtype=np.uint8)
    Y[0::2] = f[:,0]
    Y[1::2] = f[:,2]
    Y = Y.reshape(rows, cols)
    # TODO: Y is only the Y plane

    # crop image
    Y = Y[config.img_yaxis_start:config.img_yaxis_end + 1, config.img_xaxis_start:config.img_xaxis_end + 1]

    # resample image 
    Y = cv2.resize(Y, config.img_resample_dim, cv2.INTER_LINEAR)

    # convert to float
    X = np.empty((rows * cols), dtype=np.float64)
    X = Y / 127.5 - 1

    return Y, X

if __name__ == "__main__":
    cnn_config = DataConfig()

    while True:
      Y, X = read_uyvy('/tmpfs/frame.yuv', cnn_config)
      if Y is None:
        continue
      print "\r\n"
      show_img(Y)
      print X

      key = cv2.waitKey(0)

      if key == 27:
          sys.exit(0)