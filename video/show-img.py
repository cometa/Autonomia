#!/usr/bin/env python

import cv2
import numpy as np
import sys

#img = cv2.imread('./images-1481474504/img00606.jpg')
#img = cv2.imread('./images-1481474504/img00686.jpg')
img = cv2.imread('./images-1481598612/img00686.jpg')

screen_res = 320. * 2 , 240. * 2
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)
print "scale", scale
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

print "image shape", img.shape
#cv2.rectangle(img, (0, 228), (40, 239), (0, 255, 255), 1)
cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dst_rt', window_width, window_height)
cv2.imshow('dst_rt', img)

height, width = img.shape[:2]
print "image size", width, height
# telemetry is embedded in a 40x10 rectangle in the bottom left corner
telem_w = 40
telem_h = 10
# extract the rectangle with steering and throttle into a separate image
#telem = img[228:239, 0:40]
telem = img[height - telem_h - 1:height - 1, 0:telem_w - 1]

# convert to grayscale
telemgray = cv2.cvtColor(telem, cv2.COLOR_BGR2GRAY)
# threshold the image
ret, thresh = cv2.threshold(telemgray,120,255,0)
# find contours (http://answers.opencv.org/question/40329/python-valueerror-too-many-values-to-unpack/)
"""
_, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print "contours: ", len(contours)
a = np.zeros(len(contours), dtype=object)
for i in range(0, len(contours)):
     x,y,w,h = cv2.boundingRect(contours[i])
     cv2.rectangle(telem, (x,y), (x+w,y+h), (0,255,255),1)
     print "%d:" % i,x,y,w,h
     # extract a letter 
     a[i] = telemgray[y:y+h, x:x+w]


# reverse the matrix by rows
a= a[::-1] 
"""
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
  cv2.rectangle(telem, (x,y), (x+w,y+h), (0,255,255),1)
  print "%d:" % i,x,y,w,h
  # extract a letter from the thresholded grayscale image
  #a[i] = telemgray[y:y+h, x:x+w]
  a[i] = thresh[y:y+h, x:x+w]

print "---------- Numbers (left to right):"
print a


# telem image with digits bounding boxes
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)
cv2.namedWindow('dst_rt1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dst_rt1', window_width, window_height)
cv2.imshow('dst_rt1', telem)

# telem image in grayscale
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)
cv2.namedWindow('dst_rt2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dst_rt2', window_width, window_height)
cv2.imshow('dst_rt2', telemgray)

print a[0]
height, width = a[0].shape[:2]
print "digit image size", width, height
print "digit shape", a[0].shape

size = (height, width, 1)
im = np.zeros(size, np.int8)
print "im shape", im.shape, "im size", im.size

# reshape the 7x5 image in a[0] into a 7x5x1 tensor
img = np.array(a[0]).reshape((7, 5, 1))
# resize the image into a 28x28x1 tensor
im28 = np.zeros((28,28, 1), np.int8)
im28 = cv2.resize(img, (28, 28) , 0., 0., interpolation = cv2.INTER_CUBIC)
window_width = int(im28.shape[1] * 10.)
window_height = int(im28.shape[0] * 10.)
print "window size",window_width, window_height
cv2.namedWindow('im28', cv2.WINDOW_NORMAL)
cv2.resizeWindow('im28', window_width, window_height)
cv2.imshow('im28' , im28)

# change the 28x28x1 image into a flat array of 748 normalized float elements
im28flat = im28.flatten()
squarer = lambda t: t / 255.
im28norm = np.array([squarer(xi) for xi in im28flat])
print im28norm
np.savetxt('im28norm.txt', im28norm)

for i in range(0,6):
  b = np.array(a[i]).reshape((7, 5, 1))
  print b.shape, b

  window_width = int(b.shape[1] * 40.)
  window_height = int(b.shape[0] * 40.)
  print "window size",window_width, window_height
  cv2.namedWindow('digit ' + str(i), cv2.WINDOW_NORMAL)
  cv2.resizeWindow('digit '+ str(i), window_width, window_height)
  cv2.imshow('digit '+ str(i), b)

cv2.waitKey(0)
cv2.destroyAllWindows()