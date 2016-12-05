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

import time
import json
import serial
import string
import sys
import subprocess
import utils
from cometalib import CometaClient
from runtime import Runtime
from gpslib import GPS
import api
import copy

import pdb
import car

# vehicle states
IDLE = 0
RUNNING = 1
PAUSE = 2

# vehicle running modes
NORMAL = 0    # driven by RC remote control or remotely from Cometa
AUTO = 1      # fully autonomous driving

THETA_CENTER = 90
MOTOR_NEUTRAL = 90

car.state = IDLE
car.mode = NORMAL
car.cur_steering = THETA_CENTER
car.cur_throttle = MOTOR_NEUTRAL

# options flags
car.capture = False     # capturing video and telemetry for CNN training
car.streaming = False   # streaming video to cloud server

# shortcut to refer to the system log in Runtime
Runtime.init_runtime()
syslog = Runtime.syslog

config = Runtime.read_config()

arport = None

TELENAME = '/tmp/meta.txt'  # used by video streamer

def setup_arduino():
  """ Arduino Nano radio and servo controller setup. """

  try:
    # set serial non-blocking 
    port = serial.Serial(config['arduino']['serial'], config['arduino']['speed'], timeout=0.0, xonxoff=False, rtscts=False, dsrdtr=False)
    port.flushInput()
    port.flushOutput()
  except Exception as e:
    syslog (e)
    return None

  # TODO: remember the heartbeat

  # wait tha start receiving from the board
  while port.inWaiting() == 0:
    time.sleep(0.1)
  return port

def input_arduino(arport):
  """ Read a line composed of throttle and steering from the Arduino controller. """

#  global cur_steering, car.cur_throttle
  inputLine = ''
  if arport.inWaiting():
    ch = arport.read(1) 
    while ch != b'\x0A':
      inputLine += ch
      ch = arport.read(1)
    try:
      # print inputLine.decode('ISO-8859-1')
      t_in, s_in = inputLine.split()
      return int(s_in), int(t_in)
    except:
      pass
  #arport.flush()
  return car.cur_steering, car.cur_throttle

def output_arduino(arport, steering, throttle):
  """ Write steering and throttle PWM values in the [0,180] range to the Arduino controller. """

  # set steering to neutral if within an interval around 90
  steering = 90 if 88 < steering < 92 else steering
  # send a new steering PWM setting to the controller
  if steering != car.cur_steering:
    car.cur_steering = steering   # update global
    arport.write(('S %d\n' % car.cur_steering).encode('ascii'))

  # send a new throttle PWM setting to the controller
  if throttle != car.cur_throttle:
    car.cur_throttle = throttle   # update global
    arport.write(('M %d\n' % car.cur_throttle).encode('ascii'))

  #arport.flush()
  return

arport = setup_arduino()

def set_steering(val):
  """ Called by the API to set the steering """
  output_arduino(arport, val, car.cur_throttle)
  return

def main():
  # pdb.set_trace()
  global steering_in

  # Read configuration object
#  global config
#  config = Runtime.read_config()
  global arport

  if config == None:
    # error reading configuration file
    return
  verbose = config['app_params']['verbose']
  syslog("Configuration: %s" % json.dumps(config))

  # Arduino low-lever controller communication port
  if arport == None:
    syslog("Error setting up Arduino board.")
    return
  else:
    syslog("Arduino setup complete.")

  # Connect to GPS 
  if 'gps' in config:
    gps = GPS()
    ret = gps.connect(config['gps']['serial'], config['gps']['speed'])
    if ret:
      syslog("Connected to GPS.")
    else:
      syslog("Error connecting to GPS.")

  # connect to Cometa
  cometa_server = config['cometa']['server']
  cometa_port = config['cometa']['port']
  application_id = config['cometa']['app_key']
  # use the machine's MAC address as Cometa device ID
  device_id = Runtime.get_serial()

  # Instantiate a Cometa object
  com = CometaClient(cometa_server, cometa_port, application_id, config['cometa']['ssl'])
  # Set debug flag
  com.debug = config['app_params']['debug']

  # bind the message_handler() callback
  com.bind_cb(api.message_handler)

  # Attach the device to Cometa.
  ret = com.attach(device_id, "Autonomia")
  if com.error != 0:
      print "(FATAL) Error in attaching to Cometa.", com.perror()
      sys.exit(2)
  # ------------------------------------------------ #
  # print "Cometa client started.\r\ncometa_server:", cometa_server, "\r\ncometa_port:", cometa_port, "\r\napplication_id:", application_id, "\r\ndevice_id:", device_id

  # When attach is successful the server returns an object of the format:
  # {"msg":"200 OK","heartbeat":60,"timestamp":1441405206}
  try:
      ret_obj = json.loads(ret)
  except Exception, e:
      print "(FATAL) Error in parsing the message returned after attaching to Cometa. Message:", ret
      sys.exit(2)

  syslog("Device \"%s\" attached to Cometa. Server timestamp: %d" % (device_id, ret_obj['timestamp']))
  if com.debug:
      print "Server returned:", ret

  steering_in, throttle_in = car.cur_steering, car.cur_throttle
  syslog("Entering application loop.")
  last_update = 0.
  last_second = 0.
  # Application main loop.
  while True:

    # per second detection
    if 1 < time.time() - last_second:
      #print "GPS readings", gps.readings
      # update global GPS readings
      car.cur_readings = copy.deepcopy(gps.readings)
      #print cur_readings
      last_second = time.time()

    # conditional to driving using RC
    if True:
      # get inputs from RC receiver in the [0.180] range
      try:
        if arport.inWaiting():
          steering_in, throttle_in = input_arduino(arport)
      except Exception, e:
        print e,' -- serial port'
        continue

    # set steering to neutral if within an interval around 90
    steering_in = 90 if 87 <= steering_in < 92 else steering_in

    if verbose: print steering_in, throttle_in 

    if car.cur_steering == steering_in and car.cur_throttle == throttle_in:
      # like it or not we need to sleep to avoid to hog the CPU in a spin loop
      time.sleep(0.01)
      continue

    # update at 30 fps
    if 0.03337 < time.time() - last_update:
      s = ('%d %d' % (steering_in, throttle_in))
      # create metadata file for embedding steering and throttle values in the video stream
      try:
        f = open('/tmp/meta.tmp', 'w', 0)
        f.write(s)
        f.close() 
        # use mv that is a system call and not preempted
        s = '/bin/mv /tmp/meta.tmp /tmp/meta.txt'
        subprocess.check_call(s, shell=True)
      except Exception, e:
        print e
        pass
    last_update = time.time()

    # -- just a pass through as a first test

    # conditional to driving using RC
    if True:
      # set new values for throttle and steering servos
      output_arduino(arport, steering_in, throttle_in)

if __name__ == '__main__':
  main()
