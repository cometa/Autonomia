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

import serial
import threading
import pynmea2
# https://github.com/Knio/pynmea2
#
# Protocol documentation:
# http://fort21.ru/download/NMEAdescription.pdf

class GPS(object):
  """Receive data from a NMEA compatible GPS """

  def __init__(self):
    self.port = None
    self.reader = None
    self.readings = {}
    """ Current readings accessible from the `readings` instance attribute """
  
  def connect(self, device, speed):
    """Connect to a serial NMEA GPS """

    try:
      self.port = serial.Serial(device, speed)
    except Exception as e:
      print (e)
      return False

    self.reader = pynmea2.NMEAStreamReader()
    # start the reading  thread
    self.threader = threading.Thread(target=self.loop)
    self.threader.daemon = True
    self.threader.start()
    return True      
            
  def loop(self):
    """ Reader thread """

    while True:
      data = self.port.read(16)
      try:
        for msg in self.reader.next(data):
        #print(msg)
          try:
            # p = pynmea2.parse(msg)
            #print msg.sentence_type
            if msg.sentence_type == 'GGA':
              #print msg.timestamp, msg.lat, msg.lat_dir, msg.lon, msg.lon_dir
              # convert degrees,decimal minutes to decimal degrees 
              lats = msg.lat
              longs = msg.lon
              lat1 = (float(lats[2]+lats[3]+lats[4]+lats[5]+lats[6]+lats[7]+lats[8]))/60
              lat = (float(lats[0]+lats[1])+lat1)
              lon1 = (float(longs[3]+longs[4]+longs[5]+longs[6]+longs[7]+longs[8]+longs[9]))/60
              lon = (float(longs[0]+longs[1]+longs[2])+lon1)
              #print lat, lon
              self.readings['lat'] = lat
              self.readings['lat_dir'] = msg.lat_dir
              self.readings['lon'] = lon
              self.readings['lon_dir'] = msg.lon_dir
              self.readings['time'] = msg.timestamp
          except Exception, e:
            print e
      except Exception, e:
        print e            
