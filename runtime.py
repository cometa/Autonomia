#!/usr/bin/env python
"""
Low-level Board run-time support for Cometa IoT devices.

Author: Marco Graziano (marco@visiblenergy.com)
"""
__license__ = """
Copyright 2016 Visible Energy Inc. All Rights Reserved.
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

__all__ = ["Runtime"]

import time
import threading
import json
import subprocess
import os
from uuid import getnode as get_mac

# Default values
DCT_FILENAME = 'config.json'

# This module uses the following enviromental variables:
#	COMETA_APIKEY
#	COMETA_SERVER
#	COMETA_PORT

#@runtimeclass
class Runtime(object):
    """
    Run-time support for Cometa IoT Devices.
    Note: the class decorator method runs after the class is created,
    """

    # Internal system time - resolution 1Hz
    __systime = 0
    __has_systime = True

    # System status
    __status = "unknown"

    # Thread IDs
    __thtime = None     # Update __systime thread

    def __init__(self):
        """
        The Runtime object instance constructor.
        """
        pass

    # Complete class initialization after loading.
    @classmethod
    def init_runtime(klass, systime=True):
        if not systime:
            # start the update systime thread for runtimes that don't have built in timer
            if klass.__thtime == None:
                klass.__thtime = threading.Thread(target=klass._update_systime)
                klass.__thtime.daemon = True  # force to exit on SIGINT
                klass.__thtime.start()
                klass.__has_systime = systime
        klass.__status = "OK"
    # 
    #-----------------------------------------------
    #
    # Update systime every second (thread)
    @classmethod
    def _update_systime(klass):
        """
        Thread to update the system time every second. (private)
        """
        while True:
            time.sleep(1)
            klass.__systime += 1    # no need for lock

    # Force systime to a new value
    @classmethod
    def set_systime(klass, time):
        """
        Set the system time to a new value.
        """
        klass.__systime = time

    # Get current systime
    @classmethod
    def get_systime(klass):
        if klass.__has_systime:
            return int(time.time())
        else:
            return klass.__systime

    # Get host systime
    @classmethod
    def get_hostsystime(klass):
    	return int(time.time())

    # Set current status
    @classmethod
    def set_status(klass, new_status):
        klass.__status = new_status

    # Get current status
    @classmethod
    def get_status(klass):
        return klass.__status
    # 
    #-----------------------------------------------
    #

    # 
    #-----------------------------------------------
    #
    # Read config from DCT (Device Configuration Table simulated in a file)
    @classmethod
    def read_config(klass):
        # read the configuration from the DCT
        try:
            f = open(DCT_FILENAME)
            content = json.loads(f.read())['config']

            # get Cometa API Key and Server name from the environment
            if not 'app_key' in content['cometa']:
            	content['cometa']['app_key'] =  os.environ['COMETA_APIKEY']
            if not 'server' in content['cometa']:
            	content['cometa']['server'] = os.environ['COMETA_SERVER']
            if not 'port' in content['cometa']: 
            	content['cometa']['port'] = int(os.environ['COMETA_PORT'])
            return content
        except Exception as e:
            print e
            return ""

    # Read all DCT
    @classmethod
    def read_dct(klass):  
        # read the DCT
        try:
            f = open(DCT_FILENAME)
            #content = f.read().replace(u'\n', u'').replace(u'\r', u'')
            content = json.loads(f.read().replace('\n', '').replace('\r', ''))
            # get Cometa API Key and Server name from the environment
            if not 'app_key' in content['cometa']:
            	content['cometa']['app_key'] =  os.environ['COMETA_APIKEY']
            if not 'server' in content['cometa']:
            	content['cometa']['server'] = os.environ['COMETA_SERVER']
            if not 'port' in content['cometa']: 
            	content['cometa']['port'] = int(os.environ['COMETA_PORT'])         
            return content
        except Exception as e:
            return ""

    # Get device serial number
    # in Linux hosts is the least six digits of the MAC address
    @classmethod
    def get_serial(klass):
        """
        Return an hex string with the current network interface MAC address.
        """
        # TODO: testing only
        #return 'A7A7A8'

        mac = ''
        m = get_mac()
        for i in range(0, 12, 2):
        #for i in range(6, 12, 2):
            mac += ("%012X" % m)[i:i+2]
        return mac

    # Get MAC address of current network interface
    @classmethod
    def get_mac_address(klass):
        """
        Return an hex string with the current network interface MAC address.
        """
        # TODO: testing only
        #return 'A7A7A8'

        return get_mac()

    # Get info of the current network interface
    @classmethod
    def get_network_info(klass):
        # TODO: testing only
        #return {'gateway': 'N/A', 'ip_address': 'N/A'}

        # get network info
        s = subprocess.Popen('ip route', shell=True, stdout=subprocess.PIPE).stdout.read()
        reply = {}
        reply['gateway'] = s.split('default via')[-1].split()[0]
        reply['ip_address'] = s.split('src')[-1].split()[0]
        return reply

    # Simple system logger
    @classmethod
    def syslog(klass, msg, escape=False):
        """
        Simple system logger.
        """
        if escape:
            #print ("[%d] %s" % (klass.__systime, msg.replace(u'\n', u'#015').replace(u'\r', u'#012')))
            print ("[%d] %s" % (klass.get_systime(), msg.replace('\n', '#015').replace('\r', '#012')))
        else:
            print ("[%d] %s" % (klass.get_systime(), msg))

    # Get list of camera devices
    @classmethod
    def list_camera_devices(klass):
		"""
		List all video devices and their names
		"""
		videodevs = ["/dev/" + x for x in os.listdir("/dev/") if x.startswith("video") ]

		# Do ioctl dance to extract the name of the device
		import fcntl
		_IOC_NRBITS   =  8
		_IOC_TYPEBITS =  8
		_IOC_SIZEBITS = 14
		_IOC_DIRBITS  =  2

		_IOC_NRSHIFT = 0
		_IOC_TYPESHIFT =(_IOC_NRSHIFT+_IOC_NRBITS)
		_IOC_SIZESHIFT =(_IOC_TYPESHIFT+_IOC_TYPEBITS)
		_IOC_DIRSHIFT  =(_IOC_SIZESHIFT+_IOC_SIZEBITS)

		_IOC_NONE = 0
		_IOC_WRITE = 1
		_IOC_READ = 2
		def _IOC(direction,type,nr,size):
			return (((direction)  << _IOC_DIRSHIFT) |
				((type) << _IOC_TYPESHIFT) |
				((nr)   << _IOC_NRSHIFT) |
				((size) << _IOC_SIZESHIFT))
		def _IOR(type, number, size):
			return _IOC(_IOC_READ, type, number, size)
		def _IOW(type, number, size):
			return _IOC(_IOC_WRITE, type, number, size)

		sizeof_struct_v4l2_capability = (16 + 32 + 32 + 4 + 4 + 16)
		VIDIOC_QUERYCAP = _IOR(ord('V'),  0, sizeof_struct_v4l2_capability)

		import array
		import struct
		emptybuf = " " * (16 + 32 + 32 + 4 + 4 + 16) # sizeof(struct v4l2_capability)
		buf = array.array('c', emptybuf)
		cameranames = []
		for dev in videodevs:
			camera_dev = open(dev, "rw")
			camera_fd = camera_dev.fileno()
			fcntl.ioctl(camera_fd, VIDIOC_QUERYCAP, buf, 1)
			cameranames.append(buf[16:48].tostring())
#			bus_info = buf[48:80].tostring()
			camera_dev.close()

		return [videodevs, cameranames]
