# Autonomia
Cloud connected, autonomous RC car

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
