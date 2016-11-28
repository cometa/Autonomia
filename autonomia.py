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

import utils
from cometalib import CometaClient
from runtime import Runtime

import pdb

# vehicle states
IDLE = 0
RUNNING = 1
PAUSE = 2

# vehicle running modes
NORMAL = 0    # driven by RC remote control or remotely from Cometa
AUTO = 1      # fully autonomous driving

THETA_CENTER = 90
MOTOR_NEUTRAL = 90

cur_state = IDLE
cur_mode = NORMAL
cur_steering = THETA_CENTER
cur_throttle = MOTOR_NEUTRAL

# options flags
capture = False     # capturing video and telemetry for CNN training
streaming = False   # streaming video to cloud server

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

  # wait tha start receiving from the board
  while port.inWaiting() == 0:
    time.sleep(0.1)
  return port

def input_arduino(arport):
  """ Read a line composed of throttle and steering from the Arduino controller. """

  global cur_steering, cur_throttle
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

  #arport.flush()
  return

global rpc_methods
rpc_methods = (#{'name':'rexec','function':_shell}, 
               #{'name':'video_devices','function':_video_devices},
)

def message_handler(msg, msg_len):
    """
    The generic message handler for Cometa receive callback.
    Invoked every time the Cometa object receives a JSON-RPC message for this device.
    It returns the JSON-RPC result object to send back to the application that sent the request.
    The rpc_methods tuple contains the mapping of names into functions.
    """
#    pdb.set_trace()
    try:
        req = json.loads(msg)
    except:
        # the message is not a json object
        syslog("Received JSON-RPC invalid message (parse error): %s" % msg, escape=True)
        return JSON_RPC_PARSE_ERROR

    # check the message is a proper JSON-RPC message
    ret,id = utils.check_rpc_msg(req)
    if not ret:
        if id and utils.isanumber(id):
            return JSON_RPC_INVALID_PARAMS_FMT_NUM % id
        if id and isinstance(id, str):
            return JSON_RPC_INVALID_PARAMS_FMT_STR % id
        else:
            return JSON_RPC_PARSE_ERROR

    syslog("JSON-RPC: %s" % msg, escape=True)

    method = req['method']
    func = None
    # check if the method is in the registered list
    for m in rpc_methods:
        if m['name'] == method:
            func = m['function']
            break

    if func == None:
        return JSON_RPC_INVALID_REQUEST

    # call the method
    try:
        result = func(req['params'])
    except Exception as e:
        print e
        return JSON_RPC_INTERNAL_ERROR_FMT_STR % str(id)

    # build the response object
    reply = {}
    reply['jsonrpc'] = "2.0"
    reply['result'] = result
    reply['id'] = req['id']

    return json.dumps(reply)


def main():
  # pdb.set_trace()
  global cur_steering, cur_throttle, cur_state, cur_mode, capture, streaming
  
  # Read configuration object
  global config
  config = Runtime.read_config()

  if config == None:
    # error reading configuration file
    return
  verbose = config['app_params']['verbose']
  syslog("Configuration: %s" % json.dumps(config))

  # Arduino low-lever controller communication port
  arport = setup_arduino()
  if arport == None:
    return
  syslog("Arduino setup complete.")

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
  com.bind_cb(message_handler)

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

  steering_in, throttle_in = cur_steering, cur_throttle
  syslog("Entering application loop.")
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
