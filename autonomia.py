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
from runtime import Runtime

import pdb

# vehicle states
IDLE = 0
RUNNING = 1
PAUSE = 2

# vehicle running modes
NORMAL = 0    # driven by RC remote control or remotely from Cometa
AUTO = 1      # fully autonomous driving

# options flags
CAPTURE = False   # capturing video and telemetry for CNN training
STREAMING = False # streaming video to cloud server

THETA_CENTER = 90
MOTOR_NEUTRAL = 90

# shortcut to refer to the system log in Runtime
Runtime.init_runtime()
syslog = Runtime.syslog

def setup_arduino():
  """ Arduino Nano radio and servo controller setup. """

  try:
    # set serial non-blocking 
    port = serial.Serial(config['arduino']['serial'], config['arduino']['speed'], timeout=0.1, xonxoff=False, rtscts=False, dsrdtr=False)
    port.flushInput()
    port.flushOutput()
  except Exception as e:
    syslog (e)
    return None

  # TODO: remember the heartbeat
  return port

def input_arduino(arport):
    """ Read a line composed of throttle and steering from the Arduino controller. """

    inputLine = ''
    if arport.inWaiting():
      ch = arport.read(1) 
      while ch != b'\x0A':
        inputLine += ch
        ch = arport.read(1)
      try:
        # print inputLine.decode('ISO-8859-1')
        throttle_in, steering_in = inputLine.split()
        return int(steering_in), int(throttle_in)
      except:
        pass

    arport.flush()
    return cur_steering, cur_throttle

def output_arduino(arport, steering, throttle):
  """ Write steering and throttle PWM values in the [0,180] range to the Arduino controller. """

  global cur_steering, cur_throttle
  # set steering to neutral if within an interval around 90
  steering = 90 if 88 < steering < 92 else steering

  # send a new steering PWM setting to the controller
  if steering != cur_steering:
    cur_steering = steering   # update global
    arport.write(('S %d\n' % cur_steering).encode('ascii'))

  # send a new throttle PWM setting to the controller
  if throttle != cur_throttle:
    cur_throttle = throttle   # update global
    arport.write(('M %d\n' % cur_throttle).encode('ascii'))

  arport.flush()
  return

def main():
  # pdb.set_trace()
  
  # Read configuration object
  global config
  config = Runtime.read_config()

  if config == None:
    # error reading configuration file
    return
  verbose = config['app_params']['verbose']

  global cur_steering, cur_throttle

  # Arduino low-lever controller communication port
  arport = setup_arduino()
  if arport == None:
    return

  syslog("Arduino setup complete.")

#  print "State %d" % cur_state
  cur_steering, cur_throttle, steering_in, throttle_in = 90, 90, 90, 90

  # Application main loop.
  while True:

    # get inputs from RC receiver in the [0.180] range
    if arport.inWaiting():
      steering_in, throttle_in = input_arduino(arport)
      if verbose: print steering_in, throttle_in 

    # -- just a pass through as a first test
  
    # set mew values for throttle and steering servos
    output_arduino(arport, steering_in, throttle_in)

if __name__ == '__main__':
  main()
