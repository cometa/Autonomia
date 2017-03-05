""" 
Utility functions Cometa agent.

"""
__license__ = """
Copyright 2016 Visible Energy Inc. All Rights Reserved.
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

import math
import hashlib, hmac
import numpy as np
import cv2

def check_rpc_msg(req):
    ret = False
    id = None
    k = req.keys()
    # check presence of required id attribute
    if 'id' in k:
        id = req['id']
    else:
        return ret, id
    # check object length
    if (len(k) != 4):
        return ret, id
    # check presence of required attributes
    if (not 'jsonrpc' in k) or (not 'method' in k) or (not 'params' in k):
        return ret, id
    # check for version
    if req['jsonrpc'] != "2.0":
        return ret, id
    # valid request
    return True,id

def isanumber(x):
    try:
        int(x)
    except ValueError:
        try:
            float(x)
        except ValueError:
            return False
    return True

def buildKey(mac, secret):
    """Return the camera streaming key."""
    h = hmac.new(secret, mac, digestmod=hashlib.sha256).hexdigest()
    return mac + '-' + h[0:32]

# images are aquired by ffmpeg -s 320x240 -pix_fmt  yuyv422 
def read_uyvy(filename, config, rows=240, cols=320):
    # input image size 
    image_size = rows * cols * 2
    # read a YUYV raw image and extract the Y plane - YUV 4:2:2 - (Y0,U0,Y1,V0),(Y2,U2,Y3,V2)
    # this is equivalent to YUY2 pixel format http://www.fourcc.org/pixel-format/yuv-yuy2
    fd = open(filename,'rb')
    f = np.fromfile(fd, dtype=np.uint8, count=image_size)
    if len(f) != image_size: #rows*cols*2
        # error in reading
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
    Y = cv2.resize(Y, config.img_resample_dim) #, cv2.INTER_LINEAR)

    # Y is of shape (1,:,:,:)
    Y = Y.reshape(1, config.img_resample_dim[0], config.img_resample_dim[1], config.num_channels)

    # cast to float and normalize the image values
    Y_f = np.empty((rows * cols), dtype=np.float64)
    Y_f = Y / 127.5 - 1

    # reshape as a tensor for model prediction
    return Y_f

def steering2bucket(s):
    """ Convert from [0,180] range to a bucket number in the [0,14] range with log distribution to stretch the range of the buckets around 0 """
    s -= 90
    return int(round(math.copysign(math.log(abs(s) + 1, 2.0), s))) + 7

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

def throttle2bucket(t):
    """ Convert throttle from [0,180] range to a bucket number in the [0.14] range using a map. """

    for max_in_bucket,bucket in throttle_map:
        if t <= max_in_bucket:
            return bucket
    return 14

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
