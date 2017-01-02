# Autonomia
A project for a cloud connected, autonomous RC 1/10 scale electric car to participate in the [DIY Robocars self-racing-cars events.](/home/oem/Autonomia-Video/1482616994a) The project makes extensive use of [Vederly](www.cometa.io), a video and device management cloud platform for mobility applications developed by Visible Energy Inc.

## Running Modes
The Autonomia vehicle has three running modes:

1. Controlled by the radio remote control (RC)
2. Controlled remotely through the Remote JSON/RPC API and the Vederly cloud servers
3. Autonomously, driven by predictions of an end-to-end convolutional neural network (CNN)

In any running mode, the vehicle is connected to the Vederly cloud servers using the [Cometa API](http://www.cometa.io/cometa-api.html). It is also remotely managed and responding to the commands in the JSON/RPC Remote API, as well streaming telemetry and live video from the on-board camera at 30 frames per second to the Vederly cloud managemengt platform.

## Hardware
The hardware added to a commercial RC car, such as the [Trexxas Stampede](https://traxxas.com/products/models/electric/36054-1stampede?t=details) used in the initial implementation consists of:

1. [Raspberry PI 3 model B](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/)
2. [Arduino Nano](https://www.arduino.cc/en/Main/arduinoBoardNano)
3. [Logitech C920 camera](http://www.logitech.com/en-us/product/hd-pro-webcam-c920)


###Dependencies

Python:
```
$ sudo apt-get install libpython2.7-dev
```
```
$ pip install http_parser
$ pip install pynmea2
```

FFmpeg:
Add OpenMAX Integration Layer hardware acceleration support:
```
$ sudo apt-get install liboxmil-bellagio
```
Build FFmpeg from repo `https://github.com/FFmpeg/FFmpeg.git`

Enable the OMX h264 encoder that uses the GPU and add libraries to draw text:
```
$ ./configure --arch=armel --target-os=linux --enable-gpl --enable-nonfree \
  --enable-libx264 --enable-omx --enable-omx-rpi \
  --enable-libfreetype --enable-libfontconfig --enable-libfribidi

$ make
$ sudo make install
```
