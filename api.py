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

import json
import subprocess
import pdb
import utils
from runtime import Runtime

# JSON-RPC errors
JSON_RPC_PARSE_ERROR = '{"jsonrpc": "2.0","error":{"code":-32700,"message":"Parse error"},"id": null}'
JSON_RPC_INVALID_REQUEST = '{"jsonrpc": "2.0","error":{"code":-32600,"message":"Invalid Request"},"id":null}'

JSON_RPC_METHOD_NOTFOUND_FMT_STR = '{"jsonrpc":"2.0","error":{"code": -32601,"message":"Method not found"},"id": %s}'
JSON_RPC_METHOD_NOTFOUND_FMT_NUM = '{"jsonrpc":"2.0","error":{"code": -32601,"message":"Method not found"},"id": %d}'
JSON_RPC_INVALID_PARAMS_FMT_STR = '{"jsonrpc":"2.0","error":{"code": -32602,"message":"Method not found"},"id": %s}'
JSON_RPC_INVALID_PARAMS_FMT_NUM = '{"jsonrpc":"2.0","error":{"code": -32602,"message":"Method not found"},"id": %d}'
JSON_RPC_INTERNAL_ERROR_FMT_STR = '{"jsonrpc":"2.0","error":{"code": -32603,"message":"Method not found"},"id": %s}'
JSON_RPC_INTERNAL_ERROR_FMT_NUM = '{"jsonrpc":"2.0","error":{"code": -32602,"message":"Method not found"},"id": %d}'

# Vehicle object instantiated in the application module
car=None

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
      car.log("Received JSON-RPC invalid message (parse error): %s" % msg, escape=True)
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

  car.log("JSON-RPC: %s" % msg, escape=True)

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

# --------------------
# 
# RPC Methods

def _rexec(params):
  """Start a subprocess shell to execute the specified command and return its output.

  params - a one element list ["/bin/cat /etc/hosts"]
  """
  # check that params is a list
  if not isinstance(params, list) or len(params) == 0:
     return "Parameter must be a not empty list"    
  command = params[0]
  try:
      subprocess.check_call(command,shell=True)
      out = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
      return '\n' + out.decode()
  except Exception, e:
      print e
      return "{\"msg\":\"Invalid command.\"}"

def _video_devices(params):
  """List available video devices (v4l)."""
  vdevices = Runtime.list_camera_devices()
  ret = {}
  ret['devices'] = vdevices[0]
  ret['names'] = vdevices[1]
  return ret

def _set_telemetry_period(params):
  """Set telemetry period in seconds.

  params - JSON object {'period':5} 
  """   
  if type(params) is not dict or 'period' not in params.keys():
      return {"success": False}
  if params['period'] <= 0:
      return {"success": False}
  
  car.telemetry_period=params['period']
  return {"success": True}

def _get_config(params):
  """ Get configuration object """
  config = Runtime.read_config()
  config['app_params']['verbose']=car.verbose
  config['app_params']['telemetry_period']=car.telemetry_period
  return config

def _get_status(params):
  ret = {}
  ret['state'] = car.state
  ret['mode'] = car.mode
  ret['steering'] = car.steering
  ret['throttle'] = car.throttle
  ret['GPS'] = {}
  ret['GPS']['lat'] = car.readings['lat']
  ret['GPS']['lon'] = car.readings['lon']
  return ret

def _set_throttle(params):
  """ Set the throttle """
  if type(params) is not dict or 'value' not in params.keys():
      return {"success": False}

  val = params['value']
  print type(val)
  # only values in the [0.180] range
  if val < 0 or 180 < val:
      return {"success": False}
  # TODO: call autonomoia.set_throttle(val)

  car.throttle=val
  return {"success": True}  

def _set_steering(params):
  """ Set the steering """
  if type(params) is not dict or 'value' not in params.keys():
      return {"success": False}

  val = params['value']
  # only values in the [0.180] range
  if val < 0 or 180 < val:
      return {"success": False}
  #car.cur_steering = val
  car.steering=val
  return {"success": True}  

def _set_mode(params):
  """ Set the vehicle running mode"""
  if type(params) is not dict or 'value' not in params.keys():
    return {"success": False}

  val = params['value']
  if val not in ("AUTO","TRAINING"):
    return {"success": False}

  if val == "AUTO":
    car.mode2auto()
  elif val == "TRAINING":
    car.mode2training()
  return {"success": True}  

def _stop(params):
  car.state2idle()
  return {"success": True}  

def _start(params):
  car.state2run()
  return {"success": True}  

def _video_stop(params):
  return

def _video_start(params):
  return

global rpc_methods
rpc_methods = ({'name':'rexec','function':_rexec}, 
               {'name':'video_devices','function':_video_devices}, 
               {'name':'set_telemetry_period','function':_set_telemetry_period}, 
               {'name':'get_config','function':_get_config}, 
               {'name':'get_status','function':_get_status}, 
               {'name':'set_throttle','function':_set_throttle}, 
               {'name':'set_steering','function':_set_steering}, 
               {'name':'set_mode','function':_set_mode}, 
               {'name':'stop','function':_stop}, 
               {'name':'start','function':_start}, 
               {'name':'video_start','function':_video_start}, 
               {'name':'video_stop','function':_video_stop}, 
)
