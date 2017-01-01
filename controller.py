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
import Queue
import threading
import subprocess

import streamer
import utils
import numpy as np
from keras.models import model_from_json

class States:
  """ Vehicle states """
  IDLE=1
  RUNNING=2
  PAUSE=3
  STOPPED=0

class Modes:
  """ Vehicle running modes """
  AUTO=1        # fully autonomous
  TRAINING=2    # RC controlled to capture training video
  REMOTE=3      # controlled remotely through Cometa JSON/RPC 

# --- Module constansts  
THETA_CENTER = 90
MOTOR_NEUTRAL = 90
# filename to fetch telemetry from -- updated atomically by the controller loop at 30 Hz
TELEMFNAME = '/tmpfs/meta.txt'

# filename where to store the last frame -- used by the application loop and as parameter for the CNN prediction
FRAMEFNAME = '/tmpfs/frame.yuv'

def setup_arduino(config, logger):
  """ Arduino radio receiver and servos controller setup. """
  try:
    # set serial non-blocking 
    port = serial.Serial(config['arduino']['serial'], config['arduino']['speed'], timeout=0.0, xonxoff=False, rtscts=False, dsrdtr=False)
    port.flushInput()
    port.flushOutput()
  except Exception as e:
    logger("Arduino setup: %s" % e)
    return None
  # wait the board to start
  while port.inWaiting() == 0:
    time.sleep(0.1)
  return port

class RCVehicle(object):
  """ 
  Vehicle controller class
    config - configuration object from config.json
    log - system logger 
  """
  # current state and mode
  state = None
  mode = None

  # running options flags
  capture = False     # capturing video and telemetry for CNN training
  streaming = False   # streaming video to cloud server

  # current servo and motor values
  steering = None
  throttle = None

  # Arduino serial port
  arport = None

  # Mailbox for asynchronous commands
  mbox = Queue.Queue()

  # GPS readings
  readings={}

  # System logger
  log = None

  def __init__(self, config, logger):
    self.state=States.IDLE
    self.steering=THETA_CENTER
    self.throttle=MOTOR_NEUTRAL
    self.serial=config['serial']
    self.config = config

    # Set the system log
    self.log=logger
    self.verbose=config['app_params']['verbose']

    # CNN predicting model
    self.cnn_model = None

    self.arport=setup_arduino(config, self.log) 
    while self.arport == None:
      self.log("Fatal error setting up Arduino board. Cannot proceed without properly connecting to the control board.")
      time.sleep(5)
      self.arport=setup_arduino(config, self.log) 
    # TODO: remember to create a board heartbeat thread

    # Global lock
    self.glock = threading.Lock()

    # Start the main loop
    self.loop_t = threading.Thread(target=self.control_loop)
    self.loop_t.daemon = True   # force to exit on SIGINT

    self.telemetry_period=config['app_params']['telemetry_period']
    return

  def load_model(self, modelpath):
    """ Load a CNN model """
    try:
      self.model = model_from_json(open(modelpath + ".json").read())
      self.model.load_weights(modelpath + ".h5")
    except Exception, e:
      self.log("Failed to load CNN model %s. (%s)" % (modelpath,str(e)) )
      return

    self.log("Loaded CNN model %s." % modelpath)
    return

  def state2run(self):
    """ State transition to RUNNING """
    if self.state == States.RUNNING:
      return
    self.steering=THETA_CENTER
    self.throttle=MOTOR_NEUTRAL
    self.state=States.RUNNING
    self.log("State RUNNING")
    return

  def state2stopped(self):
    """ State transition to PAUSE """
    if self.state == States.STOPPED:
      return
    self.steering=THETA_CENTER
    self.throttle=MOTOR_NEUTRAL
    self.state=States.STOPPED
    self.output_arduino(self.steering, self.throttle)
    self.log("State STOPPED")    
    return

  def state2idle(self):
    """ State transition to IDLE """
    if self.state == States.IDLE:
      return
    self.steering=THETA_CENTER
    self.throttle=MOTOR_NEUTRAL
    self.state=States.IDLE
    self.output_arduino(self.steering, self.throttle)
    self.log("State IDLE")
    return

  def mode2auto(self):
    """ Mode transition to AUTO """
    if self.mode == Modes.AUTO:
      return
    # TODO: start the video fast video streamer

    self.mode=Modes.AUTO
    self.arport.flushInput()
    self.arport.flushOutput()    
    self.log("Mode AUTO")    
    return

  def mode2training(self):
    """ Mode transition to TRAINING """
    if self.mode == Modes.TRAINING:
      return
    # TODO: start the video streamer with telemetry annotations
    self.mode=Modes.TRAINING
    self.arport.flushInput()
    self.arport.flushOutput()
    self.log("Mode TRAINING")
    return

  def mode2remote(self):
    """ Mode transition to REMOTE """
    if self.mode == Modes.REMOTE:
      return

    self.arport.flushInput()
    self.arport.flushOutput()  
    self.steering=THETA_CENTER
    self.throttle=MOTOR_NEUTRAL  
    self.mode=Modes.REMOTE
    self.log("Mode REMOTE")    
    return

  def start(self):
    """ Initial start """
    self.mode2training()
    self.state2run()
    self.loop_t.start()
    return

  def telemetry(self):
    ret = {}
    ret['type'] = 1
    ret['time'] = str(int(time.time() * 1000))
    ret['device'] = self.serial
    ret['state'] = self.state
    ret['mode'] = self.mode
    ret['steering'] = self.steering
    ret['throttle'] = self.throttle
    ret['GPS'] = {}
    try:
      ret['GPS']['lat'] = int(self.readings['lat'] * 10E4) / 10E4
      ret['GPS']['lon'] = int(self.readings['lon'] * 10E4) / 10E4
    except:
      # leave GPS empty
      pass
    return ret

  def input_arduino(self):
    """ Read a line composed of throttle and steering values received from the RC. """
    inputLine = ''
    if self.arport.inWaiting():
      ch = self.arport.read(1) 
      while ch != b'\x0A':
        inputLine += ch
        ch = self.arport.read(1)
      try:
        # print inputLine.decode('ISO-8859-1')
        t_in, s_in = inputLine.split()
        # return the steering and throttle values from the receiver
        return int(s_in), int(t_in)
      except:
        pass
    # return current values after a reading error
    return self.steering, self.throttle

  def output_arduino(self, steering, throttle):
    """ Write steering and throttle PWM values in the [0,180] range to the controller. """
    # set steering to neutral if within an interval around 90
    steering = 90 if 88 < steering < 92 else steering
    # send a new steering PWM setting to the controller
    if self.mode == Modes.TRAINING:
      if steering != self.steering:
        self.steering = steering   # update global
        self.arport.write(('S %d\n' % self.steering).encode('ascii'))
    else:
        self.arport.write(('S %d\n' % self.steering).encode('ascii'))

    # send a new throttle PWM setting to the controller
    if self.mode == Modes.TRAINING:
      if throttle != self.throttle:
        self.throttle = throttle   # update global
        self.arport.write(('M %d\n' % self.throttle).encode('ascii'))
    else:
      self.arport.write(('M %d\n' % self.throttle).encode('ascii'))
    return

  # ---------------------------------------
  #
  def control_loop(self):
    """ Controller main loop """
    last_update=0
    steering_in=self.steering
    throttle_in=self.throttle
    mv = '/bin/mv /tmpfs/meta.tmp ' + TELEMFNAME

    # Load a CNN model -- must be done in the same thread of the prediction
    self.load_model(self.config['app_params']['model'])

    while True:
      now = time.time()
      #
      # ------------------------------------------------------------
      #
      if self.state == States.RUNNING and self.mode == Modes.TRAINING:
        # get inputs from RC receiver in the [0,180] range
        try:
          if self.arport.inWaiting():
            steering_in, throttle_in = self.input_arduino()
        except Exception, e:
          self.log("input_arduino: %s" % str(e))
          continue
        # set steering to neutral if within an interval around 90
        steering_in = 90 if 87 <= steering_in < 92 else steering_in
        if self.verbose: print steering_in, throttle_in 

        if self.steering == steering_in and self.throttle == throttle_in:
          # like it or not we need to sleep to avoid to hog the CPU in a spin loop
          time.sleep(0.01)
          continue

        # update telemetry file 30 times per second
        if 0.03337 < now - last_update:
          s = "%03d %03d" %  (steering_in, throttle_in)
          # create metadata file for embedding steering and throttle values in the video stream
          try:
            f = open('/tmpfs/meta.tmp', 'w', 0)
            f.write(s)
            f.close() 
            # use mv that is a system call and not preempted (atomic)
            subprocess.check_call(mv, shell=True)
          except Exception, e:
            self.log("writing file: %s" % str(e))
            pass
        last_update = now
        # set new values for throttle and steering servos
        self.output_arduino(steering_in, throttle_in)
      #
      # ------------------------------------------------------------
      #
      elif self.state == States.RUNNING and self.mode == Modes.AUTO:
        # predict steering and trhottle and set the values at a rate depending on preditiction speed
        start_t = time.time()
        Y = utils.read_uyvy(FRAMEFNAME) # Y is of shape (1,240,320,1)
        if Y is None:
          print "image not acquired"
          continue
        # normalize the image values
        Y = Y / 255.
        # predict steering and throttle
        p = self.model.predict(Y[0:1])
        self.steering = np.argmax(p[:, :15],  1)[0]
        self.throttle = np.argmax(p[:, 15:], 1)[0]
        self.steering = utils.bucket2steering(self.steering)
        self.throttle = utils.bucket2throttle(self.throttle)
        # clip the prediction for testing
        if self.throttle > 110:
          self.throttle = 110
        if self.throttle < 80:
          self.throttle = 80
        print self.steering, self.throttle
        self.output_arduino(self.steering, self.throttle)
        dt = time.time() - start_t
        time.sleep(0.001) #0.034 - max(0.033, dt))
        print "execution time:", dt
      #
      # ------------------------------------------------------------
      #
      elif self.state == States.RUNNING and self.mode == Modes.REMOTE:
        self.output_arduino(self.steering, self.throttle)
        time.sleep(0.2)
      #
      # ------------------------------------------------------------
      #
      elif self.state == States.IDLE:
        time.sleep(0.1)
      #
      # ------------------------------------------------------------
      #
      else:
        time.sleep(1)
