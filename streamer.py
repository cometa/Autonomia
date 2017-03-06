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

import os
import time
import subprocess as sp
import numpy as np
import cv2

# filename to fetch telemetry from -- updated atomically by the controller loop at 30 Hz
TELEMFNAME = '/tmpfs/meta.txt'

# filename where to store the last frame -- used by the application loop and as parameter for the CNN prediction
FRAMEFNAME = '/tmpfs/frame.yuv'

log=None
config=None
camera=None

# Vehicle object instantiated in the application module
car=None

streaming=False

def init(conf, logger):
  global config, log, camera

  config=conf
  log=logger

  videodevs = ["/dev/" + x for x in os.listdir("/dev/") if x.startswith("video") ]
  if len(videodevs) == 0:
    log("Fatal error. Cannot proceed without cameras connected.")
    return None

  print videodevs
  # create an empty telemetry file
  s = 'echo > ' + TELEMFNAME 
  sp.check_call(s, shell=True)

  # use the first camera
  camera=videodevs[0]

  # set resolution and encoding (Logitech C920)
  s = 'v4l2-ctl --device=' + camera + ' --set-fmt-video=width=320,height=240,pixelformat=1'
  log("Setting camera: %s" % s)
  # execute and wait for completion
  sp.check_call(s, shell=True)

  return camera

def video_stop():
  """ Stop running video capture streamer """

  streaming = True
  return

  try:  
    pname = config['video']['streamer']
  except:
    log("Error cannot stop the video streamer. Streamer not defined in config.json")
    return None  

  s = 'killall ' + pname
  FNULL = open(os.devnull, 'w')
  try:
    # execute and wait for completion
    sp.check_call(s, shell=True, stderr=FNULL) 
  except Exception, e:
    # fails when no ffmpeg is running
    if config['app_params']['verbose']: 
      log("Error stopping streamer. %s" % e)
    else:
      pass
  return

def __video_start(telem):
  """ Start a video streamer """ 
  global config, log, camera

  # insure no streamer is running
  video_stop()
  time.sleep(1)

  try:  
    pname = config['video']['streamer']
  except:
    log("Error cannot start the video streamer. Streamer not defined in config.json")
    return None

  # set video codec depending on platform
  vcodec = 'h264_omx' if 'raspberrypi' in os.uname() else 'h264'
  # to suppress output when running the streamer
  FNULL = open(os.devnull, 'w')
  if telem:
    # streaming video with embedded telemetry
    params = [pname, '-r','30', '-use_wallclock_as_timestamps', '1', '-thread_queue_size', '512', '-f', 'v4l2', '-i', camera, '-c:v ', vcodec, '-maxrate', '768k', '-bufsize', '960k']
    format = 'format=yuv444p,drawbox=y=ih-h:color=black@0.9:width=40:height=12:t=max,drawtext=fontfile=OpenSans-Regular.ttf:textfile=' + TELEMFNAME + ':reload=1:fontsize=10:fontcolor=white:x=0:y=(h-th-2),format=yuv420p'
    url = 'rtmp://' + config['video']['server'] + ':' + config['video']['port'] + '/src/' + config['video']['key']
    params = params + ['-vf', format, '-threads', '4', '-r', '30', '-g', '60', '-f', 'flv', url]
    # spawn a process and do not wait
    pid = sp.Popen(params, stderr=FNULL)
  else:
    # streaming video and saving the last frame for CNN prediction
    params = [pname, '-r','30', '-use_wallclock_as_timestamps', '1', '-thread_queue_size', '512', '-f', 'v4l2', '-i', camera, '-c:v ', vcodec, '-maxrate', '768k', '-bufsize', '960k']
    url = 'rtmp://' + config['video']['server'] + ':' + config['video']['port'] + '/src/' + config['video']['key']
    params = params + ['-threads', '4', '-r', '30', '-g', '60', '-f', 'flv', url]
    params = params + ['-vcodec', 'rawvideo', '-an', '-updatefirst', '1', '-y', '-f', 'image2', FRAMEFNAME]
    # to transcode use format YUYV422:
    # $ ffmpeg -vcodec rawvideo -s 320x240 -r 1 -pix_fmt  yuyv422  -i frame.yuv rawframe.jpg
    # spawn a process and do not wait
    pid = sp.Popen(params, stderr=FNULL)
  return pid

def video_start(telem):

  try:  
    pname = config['video']['streamer']
  except:
    log("Error cannot start the video streamer. Streamer not defined in config.json")
    return None

  i_command = [ pname,
            '-r', '30',
            '-use_wallclock_as_timestamps', '1',
            '-f', 'v4l2',
            '-i', '/dev/video0',
            '-vb','1000k',
            '-f', 'image2pipe',
            '-pix_fmt', 'yuyv422',
            '-vcodec', 'rawvideo', '-']
  i_pipe = sp.Popen(i_command, stdout = sp.PIPE, bufsize=10**5)

  url = 'rtmp://' + config['video']['server'] + ':' + config['video']['port'] + '/src/' + config['video']['key']

  o_command = [ pname,
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '320x240', # size of one frame
#         '-pix_fmt', 'rgb24',
        '-pix_fmt', 'rgb24', #'yuyv422', rgb24
        '-r', '30', # frames per second
        '-i', 'pipe:0', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-c:v','libx264',
        '-profile:v','main',
        '-preset','ultrafast',
        '-pix_fmt', 'yuv420p',
        '-b:v','1000k',
        '-bufsize','2000k',
        '-g','30',
        '-f','flv',
        url ]
  o_pipe = sp.Popen(o_command, stdin=sp.PIPE, stderr=sp.PIPE)

  width = 320 # 640
  height = 240 #480

  rows = height
  cols = width

  image_size = rows * cols * 2 # *3

  streaming = True
  while streaming:
    raw_image = i_pipe.stdout.read(image_size)

    if telem:
      msg = car.telemetry()
      car.com.send_data(json.dumps(msg))    

    f = np.fromstring(raw_image, dtype=np.uint8)
    i_pipe.stdout.flush()

    img = f.reshape(rows, cols, 2)  # 2)

    # convert to RGB
    rgb_img =  cv2.cvtColor(img, cv2.COLOR_YUV2RGB_YUY2)  # working w camera format yuyv422

 #   rgb_img = img
 #   rgb_img =  cv2.cvtColor(img, cv2.COLOR_YUV420p2RGB) 

    # draw a center rectangle
    cv2.rectangle(rgb_img,(130,100),(190,140),(255,0,0),2) 
  #  M = cv2.getRotationMatrix2D((width/2,height/2),180,1)

    # rotate the image 90 degrees twice and bring back to normal
  #  dst = cv2.warpAffine(rgb_img,M,(width,height))

    rgb_img[0,0,0] = car.steering 
    rgb_img[0,0,1] = car.throttle
    # output the image
    o_pipe.stdin.write(rgb_img.tostring())

