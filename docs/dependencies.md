### Dependencies

Python:
```
$ sudo apt-get install libpython2.7-dev
```
```
$ pip install http_parser
$ pip install pynmea2

$pip install tensorflow
$pip install keras
$ sudo apt-get install python-opencv
$ pip install h5py

```
FFmpeg:
Add `OpenMAX Integration Layer` hardware acceleration support:
```
$ sudo apt-get install liboxmil-bellagio
or
$ sudo apt-get install liboxmil-bellagio-dev
```
Install fontconfig and libfribidi:
```
$ sudo apt-get install fontconfig
$ sudo apt-get install libfontconfig1-dev

$ sudo apt-get install libfribidi0
$ sudo apt-get install libfribidi-dev
```
Install libx264:
```
$ sudo apt-get install libx264-142 libx264-dev
```
Build `FFmpeg` from repo `https://github.com/FFmpeg/FFmpeg.git`

Enable the `OMX h264` encoder that uses the GPU and add libraries to draw text:
```
$ ./configure --arch=armel --target-os=linux --enable-gpl --enable-nonfree \
  --enable-libx264 --enable-omx --enable-omx-rpi \
  --enable-libfreetype --enable-libfontconfig --enable-libfribidi

$ make
$ sudo make install
```
For data preparation `OpenCV2` is needed.

For training and model evaluation in the application `Keras` and `Tensorflow` are needed. On the Raspberry PI, `OpenCV2` is not needed.

### Cloud Connection
The Autonomia application has a dependency on Vederly, a video and device management cloud platform for mobility applications, including a two-way message broker for device-to-cloud and cloud-to-device secure communication. 

The main application manages the connection to the Vederly server as defined in the `config.json` parameters file, streams video using RTMP to the server, and exposes methods for JSON-RPC remote procedure calls to the vehicle. 

If you are interested in receiving beta tester credentials and access to a Vederly cloud server for testing the Autonomia software or the cloud API send an email to cometa@visiblenergy.com
