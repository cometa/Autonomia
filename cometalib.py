"""
Author:  Emile Camus
"""
__license__ = """
Copyright 2015 Visible Energy Inc. All Rights Reserved.
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
__all__ = ["CometaClient"]

import socket
import select
import time
import threading
import ssl
# From http-parser (0.8.3)
# pip install http-parser
from http_parser.parser import HttpParser
import pdb

class CometaClient(object):
  """Connect a device to the Cometa infrastructure"""
  errors = {0:'ok', 1:'timeout', 2:'network error', 3:'protocol error', 4:'authorization error', 5:'wrong parameters', 9:'internal error'} 

  def __init__(self,server, port, application_id, use_ssl, logger):
    """
    The Cometa instance constructor.

    server: the Cometa server FQDN
    port: the Cometa server port
    application_id: the Cometa application ID
    """
    self.error = 9
    self.debug = False

    self._server = server
    self._port = port
    self._app_id = application_id
    self._use_ssl = use_ssl
    self._message_cb = None

    self._device_id = ""
    self._platform = ""
    self._hparser = None
    self._sock = None #socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._heartbeat_rate = 60
    self._trecv = None
    self._thbeat = None
    self._hb_lock = threading.Lock()
    self._reconnecting = False
    self.log = logger
    return

  def attach(self, device_id, device_info):
    """
    Attach the specified device to a Cometa registered application. 
    Authentication is done using only the application_id (one-way authentication).

    device_id: the device unique identifier
    device_info: a description of the platform or the device (used only as a comment)
    """
    self._device_id = device_id
    self._platform = device_info
    self._hparser = HttpParser()
    tsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if self._use_ssl:
      self._sock = ssl.wrap_socket(tsock, ssl_version=ssl.PROTOCOL_SSLv23,  ciphers="AES256-GCM-SHA384")
    else:
      self._sock = tsock
    try:
      self._sock.connect((self._server, self._port))
      sendBuf="POST /v1/applications/%s/devices/%s HTTP/1.1\r\nHost: api.cometa.io\r\nContent-Length:%d\r\n\r\n%s" % (self._app_id,device_id,len(device_info),device_info)
      self._sock.send(sendBuf)
      recvBuf = ""
      while True:
        data = self._sock.recv(1024)
        if not data:
          break

        dataLen = len(data)
        nparsed = self._hparser.execute(data, dataLen)
        assert nparsed == dataLen

        if self._hparser.is_headers_complete():
          if self.debug:
            print "connection for device %s headers received" % (device_id)
            print self._hparser.get_headers()

        if self._hparser.is_partial_body():
          recvBuf = self._hparser.recv_body()
          if self.debug:
            print "connection for device %s body received" % (device_id)
            print recvBuf         
          #TODO: check for error in connecting, i.e. 403 already connected

          # reading the attach complete message from the server  
          # i.e. {"msg":"200 OK","heartbeat":60,"timestamp":1441382935}
          if len(recvBuf) < 16 or recvBuf[1:12] != '"msg":"200"':
            self.error = 5
            print "Error in string from server; %s" % recvBuf
            return recvBuf

          # reset error
          self.error = 0

          # set the socket non blocking
          self._sock.setblocking(0) 

          # do not (re)start the threads during a reconnection
          if self._reconnecting:
            self._reconnecting = False
            return recvBuf

          if self.debug:
            print "connection for device %s completed" % (device_id)
                      # start the hearbeat thread
          self._thbeat = threading.Thread(target=self._heartbeat)
          self._thbeat.daemon = True
          self._thbeat.start()
            
          # start the receive thread
          #time.sleep(2)
          self._trecv = threading.Thread(target=self._receive)
          self._trecv.daemon = True # force to exit on SIGINT
          self._trecv.start()


          return recvBuf
    except Exception, e:
      print e
      self.error = 2
      return

  def send_data(self, msg):
    """
    Send a data event message upstream to the Cometa server.
    If a Webhook is specified for the Application in the Cometa configuration file /etc/cometa.conf on the server, 
    the message is relayed to the Webhook. Also, the Cometa server propagates the message to all open devices Websockets. 
    """
    sendBuf = "%x\r\n%c%s\r\n" % (len(msg) + 1,'\07',msg)
    if self._reconnecting:
      if self.debug:
        print "Error in Cometa.send_data(): device is reconnecting."
      return -1
    try:
      self._hb_lock.acquire()
      self._sock.send(sendBuf)
      self._hb_lock.release()     
    except Exception, e:
      if self.debug:
        print "Error in Cometa.send_data(): socket write failed."
      return -1
    return 0

  def bind_cb(self, message_cb):
    """
    Binds the specified user callback to the Cometa instance.
    """
    self._message_cb = message_cb
    return

  def perror(self):
    """
    Return a string for the current error.
    """
    return CometaClient.errors[self.error]

  def _heartbeat(self):
    """
    The heartbeat thread.
    The hearbeat message is a chunk of length 3 with the MSG_HEARBEAT byte and closed with CRLF.
    This thread detects a server disconnection and attempts to reconnect to the Cometa server.
    """
    if self.debug:
      print "Hearbeat thread started.\r"
        
    while True:
      time.sleep(self._heartbeat_rate)
      if self._reconnecting:
        print "--- heartbeat while reconnecting"
        continue  
      sendBuf = "1\r\n%c\r\n" % '\06'
      self.log("sending heartbeat")
      try:
        self._hb_lock.acquire()
        self._sock.send(sendBuf)
        self._hb_lock.release()
      except Exception, e:
        print "--- error sending heartbeat"
        return

  def _receive(self):
    """
    The receive and user callback dispatch loop thread.
    """
    if self.debug:
      print "Receive thread started.\r"
    while True:
      ready_to_read, ready_to_write, in_error = select.select([self._sock.fileno()],[],[self._sock.fileno()], 15)

      # check for timeout
      if not (ready_to_read or ready_to_write or in_error):
        continue

      for i in in_error:
        # handle errors as disconnections and try to reconnect to the server
        print "Network error in receive loop (error). Reconnecting..."
        self._sock.close()
        # self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._reconnecting = True
        ret = self.attach(self._device_id, self._platform)
        if self.error != 0:
          print "Error in attaching to Cometa.", self.perror()
          time.sleep(15)
          continue
        else:
          print "Device attached to Cometa.", ret
        continue

      data = None
      for i in ready_to_read:
        try:
          data = self._sock.recv(1024)
        except Exception, e:
          print e
          pass

      if not data:
        # handle errors as disconnections and try to reconnect to the server
        print "Network error in receive loop (no data). Reconnecting..."
        try:
          self._sock.close()
        except Exception, e:
          print "--- exception in close socket."
          pass
        
        # self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._reconnecting = True
        ret = self.attach(self._device_id, self._platform)
        if self.error != 0:
          print "Error in attaching to Cometa.", self.perror()
          time.sleep(15) 
          continue
        else:
          print "Device attached to Cometa.", ret       
        continue

      if self.debug:
        print "** received: %s (%d)" % (data, len(data))      
      self._hparser.execute(data, len(data))
      if self._hparser.is_partial_body():
        to_send = self._hparser.recv_body()
        # pdb.set_trace()
        # the payload contains a HTTP chunk
        if self._message_cb:
          # invoke the user callback 
          reply = self._message_cb(to_send, len(to_send))
        else:
          reply = ""
        if self.debug:
          print "After callback."
      else:
        continue

      if self.debug:
        print "Returning result."
      sendBuf = "%x\r\n%s\r\n" % (len(reply),reply)
      try:
        self._hb_lock.acquire()
        self._sock.send(sendBuf)
        self._hb_lock.release()
      except Exception, e:
        print "--- error sending reply"
        pass

      msg = ""    
