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
import copy
import signal
import subprocess
import pdb

# ---------------------
import utils
from cometalib import CometaClient
from runtime import Runtime
from gpslib import GPS
import api
from controller import RCVehicle
import streamer

TELEMFNAME = '/tmpfs/meta.txt'

def signal_handler(signum, frame):
    sys.exit(0)

def main(argv):
  signal.signal(signal.SIGINT, signal_handler)

  Runtime.init_runtime()
  syslog = Runtime.syslog

  # Read configuration
  config = Runtime.read_config()
  if config == None:
    # error reading configuration file
    syslog("(FATAL) Error reading configuration file. Exiting.")
    return
  syslog("Configuration: %s" % json.dumps(config))

  # Connect to GPS 
  if 'gps' in config:
    gps = GPS(syslog)
    ret = gps.connect(config['gps']['serial'], config['gps']['speed'])
    if ret:
      syslog("Connected to GPS.")
    else:
      gps = None
      syslog("Error connecting to GPS on % s. Disabling." % config['gps']['serial'])

  # Connect the device to Cometa
  cometa_server = config['cometa']['server']
  cometa_port = config['cometa']['port']
  application_id = config['cometa']['app_key']
  # use the machine's MAC address as Cometa device ID
  device_id = Runtime.get_serial()
  config['serial'] = device_id
  # override camera key with new format
  config['video']['key'] = utils.buildKey(device_id, str(application_id) + ':' + '1'

  # Instantiate a Cometa object
  com = CometaClient(cometa_server, cometa_port, application_id, config['cometa']['ssl'], syslog)
  com.debug = config['app_params']['debug']
  # bind the message_handler() callback
  com.bind_cb(api.message_handler)

  # Attach the device to Cometa
  connected = False

  while not connected:
    ret = com.attach(device_id, "ROV")
    if com.error != 0:
        print "Error in attaching to Cometa. Retrying ...", com.perror()
        time.sleep(1)
        continue
    # Get the timestamp from the server
    try:
        ret_obj = json.loads(ret)
    except Exception, e:
        print "Error in parsing the message returned after attaching to Cometa. Message:", ret
        time.sleep(1)
        continue
    connected = True

  # The server returns an object like: {"msg":"200 OK","heartbeat":60,"timestamp":1441405206}
  syslog("Device \"%s\" attached to Cometa. Server timestamp: %d" % (device_id, ret_obj['timestamp']))
  if com.debug:
      print "Server returned:", ret

  # Create a car controller object
  car = RCVehicle(config, syslog)

  # Initialize camera streamer
  streamer.init(config, syslog, car)

  # Start the vehicle with default training mode 
  car.start()
  car.com = com

  # Export the vechicle object to the API module
  api.car = car

  gps = None
  last_second, last_telemetry = 0., 0.
  while car.state:
    now = time.time()

    # Per second loop
    if 1 < now - last_second:
      if car.verbose and gps: print "GPS readings", gps.readings
      # update GPS readings
      try:
        if gps: car.readings = gps.readings
      except:
        pass
      last_second = now

    # Send telemetry data
#    if car.telemetry_period < now - last_telemetry: 
#      msg = car.telemetry()
#      if com.send_data(json.dumps(msg)) < 0:
#          syslog("Error in sending telemetry data.")
#      else:
#          if car.verbose:
#              syslog("Sending telemetry data %s " % msg)
#      last_telemetry = now

    time.sleep(1)

if __name__ == '__main__':
  main(sys.argv[1:])
