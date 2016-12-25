
## Video Acquisition 
Video is streamed from the car to the Vederly Cloud server using the RTMP protocol at 30 frames per second. The streaming is controlled by the `video_start` and `video_stop` remote methods.

To acquire video for CNN training, the `video_start` method must be called with the `{'telem':true}` parameter. The video generated embeds steering and throttle servo values in the bottom left of each frame. 

Ingested video is stored in the Vederly server in `flv` files of maximum 5 minutes length.
## Preparation Steps
Once downloaded a video file, the preparation steps result in two `numpy` arrays, one for the video frames and one for the corresponding steering and throttle values.

A video file has a name that includes a `camera_key` and a `timestamp`. For instance: `74DA388EAC61-1482024251.flv`

### Split video file 
A video file is split in individual frames using the `ffmpeg` command:
```
ffmpeg -i {INPUT} img%05d.jpg
```
Each frame is a file with a name containing sequential digits in order of time.
1. Create a directory to contain the frames using the timestamp in the filename:
    ```
    mkdir images-1482024251
    ```
2. Change directory and run the `ffmpeg` command: 
    ```
    cd images-1482024251
    ffmpeg -i ../74DA388EAC61-1482024251.flv -qscale:v 2 img%05d.jpg
    ```
