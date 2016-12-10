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
import subprocess

# filename to fetch telemetry from -- updated atomically by the controller loop at 30 Hz
TELEMFNAME = '/tmpfs/meta.txt'

# filename where to store the last frame -- used by the application loop and as parameter for the CNN prediction
FRAMEFNAME = '/tmpfs/frame.yuv'

log=None
config=None
camera=None

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
  subprocess.check_call(s, shell=True)

  # use the first camera
  camera=videodevs[0]

  # set resolution and encoding (Logitech C920)
  s = 'v4l2-ctl --device=' + camera + ' --set-fmt-video=width=320,height=240,pixelformat=1'
  log("Setting camera: %s" % s)
  # execute and wait for completion
  subprocess.check_call(s, shell=True)

  return camera

def video_stop():
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
    subprocess.check_call(s, shell=True, stderr=FNULL) 
  except Exception, e:
    # fails when no ffmpeg is running
    if config['app_params']['verbose']: 
      log("Error stopping streamer. %s" % e)
    else:
      pass
  return

def video_start(telem):
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
    pid = subprocess.Popen(params, stderr=FNULL)
  else:
    # streaming video and saving the last frame for CNN prediction
    params = [pname, '-r','30', '-use_wallclock_as_timestamps', '1', '-thread_queue_size', '512', '-f', 'v4l2', '-i', camera, '-c:v ', vcodec, '-maxrate', '768k', '-bufsize', '960k']
    url = 'rtmp://' + config['video']['server'] + ':' + config['video']['port'] + '/src/' + config['video']['key']
    params = params + ['-threads', '4', '-r', '30', '-g', '60', '-f', 'flv', url]
    params = params + ['-vcodec', 'rawvideo', '-an', '-updatefirst', '1', '-y', '-f', 'image2', FRAMEFNAME]
    # spawn a process and do not wait
    pid = subprocess.Popen(params, stderr=FNULL)
  return pid
