## Video Acquisition 

Video is streamed from the car to the Vederly Cloud server using the RTMP protocol at 30 frames per second, with a 240x320 resolution and H.264 encoding. The streaming is controlled by the `video_start` and `video_stop` JSON/RPC methods.

To acquire video for CNN training, the `video_start` method must be called with the `{'telem':true}` parameter. With this option, the video generated embeds steering and throttle servo values in the bottom left of each frame. 

> `steering` and `trhottle` values are in the `[0,180]` range with `90` as neutral

Ingested video is stored in the Vederly server in `flv` files of 5 minutes length with the filename including the vehicle identifier and an Epoch timestamp in seconds.

## Preparation Steps
Once downloaded a video file, the preparation steps start with exactring steering and throttle values from each frame. 

>A video file has a name that includes a `camera_key` and a `timestamp`. For instance: `74DA388EAC61-1482024251.flv`

### Extract Frames from Video
A video file is split in individual frames using the `ffmpeg` command:
```
ffmpeg -i {INPUT} -qscale:v 2 img%05d.jpg
```
> the `qscale:v 2` option is to obtain JPEG images with the best possible quality

Each frame is a file with a name containing sequential numbers in order of time.
1. Create a directory to contain the frames using the timestamp in the filename:
    ```
    mkdir images-1482024251
    ```
2. Change directory and run the `ffmpeg` command: 
    ```
    cd images-1482024251
    ffmpeg -i ../74DA388EAC61-1482024251.flv -qscale:v 2  img%05d.jpg
    ```
> At 30 fps acquisition rate, every minute of video results in 1800 still images

### Extract Telemetry from Frames

Steering and throttle are embedded in each frame in a 40x10 pixel box in the bottom left of the image as three-digit numbers, using a fixed-size font. Each digit is a 7x5 image.

>All offline image processing and CNN training, is not done on the Raspberry PI but on a cloud server or desktop.

A CNN coded in Keras/Tensorflow has been trained to recognize each digit in the telemetry box. The code for the CNN, togheter with the `digit_cnn.json` Keras model and the `digit_cnn.h5` weights, is in the `DigitsNet` directory. About 4,000 images have been used for the training. Training images and their labels are in the files `testimages.npy` and `testlabels.npy`

The `extract_telemetry.py` Python script is used to extract telemetry from all the still images in a directory, and creating the `labels.csv` file in the same directory:
```
extract_telemetry.py {DIRECTORY}
```
For instance:
```
extract_telemetry.py  images-1482024251
```
At completion, the `images-1482024251/labels.csv` contains:
```
img00001.jpg,90,89
img00002.jpg,90,89
img00003.jpg,97,89
img00004.jpg,102,100
img00005.jpg,101,99
img00006.jpg,101,99
img00007.jpg,100,98
....
```
>Embedding the telemetry into the frame by the streamer, insures almost perfect synchronization of values with the video and it is performed while streaming video, not single frames, to the Vederly Cloud for live view as well as recording.
