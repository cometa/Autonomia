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
import json
import time
import threading

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
video_thread=None

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

def video_start(telem):
  """
  The video is streamed with a pipeline:
     camera -> ffmpeg to stdout -> each frame is read and processed -> stdin into ffmpeg for RTMP push to server
  the streaming pipeline runs in a separate thread   
  """
  global video_thread

  video_thread = threading.Thread(target=video_pipe,args=(telem,))
#  video_thread.streaming = True
  video_thread.start()
  return

def video_pipe(telem):    
  global video_thread

  try:  
    pname = config['video']['streamer']
  except:
    log("Error cannot start the video streamer. Streamer not defined in config.json")
    return None

  # input ffmpeg command
  i_command = [ pname,
            '-r', '30',
            '-use_wallclock_as_timestamps', '1',
            '-f', 'v4l2',
            '-i', '/dev/video0',
            '-vb','1000k',
            '-f', 'image2pipe',
            '-pix_fmt', 'yuyv422',
            '-vcodec', 'rawvideo', '-']
  # ffmpeg stdout into a pipe
  i_pipe = sp.Popen(i_command, stdout = sp.PIPE, bufsize=10**5)

  # output ffmpeg push to the RTMP server
  url = 'rtmp://' + config['video']['server'] + ':' + config['video']['port'] + '/src/' + config['video']['key']
  o_command = [ pname,
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '320x240', # size of one frame
        '-pix_fmt', 'rgb24', #'yuyv422', rgb24
        '-r', '30', # frames per second
        '-i', 'pipe:0', # The imput comes from a pipe
# to use with standard h264 codec
#        '-an', # Tells FFMPEG not to expect any audio
#        '-c:v','libx264',
#        '-profile:v','main',
#        '-preset','ultrafast',
#        '-pix_fmt', 'yuv420p',
#        '-b:v','1000k',
# RPI GPU codec
        '-c:v', 'h264_omx',
        '-maxrate','768k',
        '-bufsize','2000k',
        '-r', '30', # frames per second
        '-g','60',
        '-f','flv',
        url ]
  # ffmpeg stdin from a pipe
  o_pipe = sp.Popen(o_command, stdin=sp.PIPE, stderr=sp.PIPE)

  # frame size
  rows = car.rows
  cols = car.cols
  image_size = rows * cols * 2 # *3

  # frame counter -- image frame filenames will have this number once the stored video file is split with ffmpeg
  count = 1
  #video_thread = threading.currentThread()

  # telemetry object sent for every frame
  ret = {}
  ret['device_id'] = car.serial   # constant for every frame

  # streaming loop
  while getattr(video_thread, "streaming", True):
    # read frame bytes from ffmpeg input
    raw_image = i_pipe.stdout.read(image_size)

    # send telemetry in the capture loop for optimal synchronization
    if telem:
      now = int(time.time() * 1000)
      ret['time'] = str(now)
      ret['s'] = car.steering
      ret['t'] = car.throttle
      ret['c'] = count
      car.com.send_data(json.dumps(ret))

    # prepare the frame for any processing
    f = np.fromstring(raw_image, dtype=np.uint8)
    img = f.reshape(rows, cols, 2)  # 2)

    # lock the frame for use in the controller loop 
    car.glock.acquire()

    # convert to RGB and assign to car object attribute
    car.frame =  cv2.cvtColor(img, cv2.COLOR_YUV2RGB_YUY2)  # working w camera format yuyv422

    # --- TEST only
    # draw a center rectangle
    # cv2.rectangle(car.frame,(130,100),(190,140),(255,0,0),2) 
    # M = cv2.getRotationMatrix2D((width/2,height/2),180,1)
    # rotate the image 90 degrees twice and bring back to normal
    #dst = cv2.warpAffine(car.frame,M,(width,height))

    # print steering and throttle value into the image (for telemetry checking only)
    s = "%04d: %03d %03d" %  (count, car.steering, car.throttle)
    cv2.putText(car.frame, s,(5,10), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,255,0), 1) 

    # release the frame lock
    car.glock.release()

    # frame counter
    count += 1

    # output the frame to the ffmpeg output process
    o_pipe.stdin.write(car.frame.tostring())
    # flush the input buffer
    i_pipe.stdout.flush()
  print 'exiting streaming loop'
  
  return


def video_stop():
  """ Stop running video capture streamer """
  global video_thread
  video_thread.streaming = False
  # wait for the thread to finish
  video_thread.join(5)

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
    log("Error stopping streamer. %s" % e)
  return

#------------------------------------------------------------------

#old version video_start
def Xvideo_start(telem):
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

# old video stop
def Xvideo_stop():
  """ Stop running video capture streamer """
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

"""
ffmpeg -i ../1488768815.flv  -vcodec rawvideo -pix_fmt yuyv422 -f image2  %03d.raw

f=open('021.raw','rb')
i=f.read(240 * 320 * 2)
ff = np.fromstring(i,dtype=np.uint8)
img = ff.reshape(240, 320, 2)
img[0,0,0]
img[0,0,1]

pass through
./ffmpeg  -r 30 -use_wallclock_as_timestamps 1  -thread_queue_size 512  -f v4l2 -vcodec h264 -i /dev/video0 -vcodec copy -f flv rtmp://stream.cometa.io:12345/src/74DA388EAC61

ffmpeg  -r 30 -use_wallclock_as_timestamps 1  -thread_queue_size 512  -f v4l2  -i /dev/video0 -c:v h264_omx -r 30 -g 60 -f flv rtmp://stream.cometa.io:12345/src/74DA388EAC61

  i_command = [ pname,
            '-r', '30',
            '-use_wallclock_as_timestamps', '1',
            '-f', 'v4l2',
            '-i', '/dev/video0',
            '-vb','1000k',
            '-f', 'image2pipe',
            '-']

  o_command = [ pname,
        '-f', 'image2pipe',
        '-i', 'pipe:0', # The imput comes from a pipe
        '-f','flv',
        url ]
    f = np.fromstring(raw_image, dtype=np.uint8)


    o_pipe.stdin.write(f.tostring())
    i_pipe.stdout.flush()
    continue
"""
