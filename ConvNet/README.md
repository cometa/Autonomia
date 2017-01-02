## Video Acquisition 

Video is streamed from the car to the Vederly Cloud server using the RTMP protocol at 30 frames per second, with a 240x320 resolution and H.264 encoding. The streaming is controlled by the `video_start` and `video_stop` JSON/RPC methods.

To acquire video for CNN training, the `video_start` method must be called with the `{'telem':true}` parameter. With this option, the video generated embeds steering and throttle servo values in the bottom left of each frame. 

>Key to keeping the car additional hardware minimal is to perform all offline image processing and CNN training not on the car computer but on a cloud server or a desktop.

**Example of image with telemetry:**

![Image with telemetry](../master/docsimg00455.jpg "image with telemetry")

> `steering` and `trhottle` values are in the `[0,180]` range with `90` as neutral. This is the values received by the RC radio receiver and captured by the Arduino controller.

Streaming video is automatically ingested and stored in the Vederly server in `flv` files of 5 minutes length with the filename including the vehicle identifier and an Epoch timestamp in seconds. A video file stored in Vederly cloud has a name that includes a `camera_key` unique for the vehicle, and an Epoch `timestamp`. For instance: `74DA388EAC61-1482024251.flv`

## Preparation Steps
Once a telemetry video file has been downloaded, the preparation steps are:

1. Extract frames from the video
2. Obtain steering and throttle label from each frame
3. Prepare the NumPy arrays with images and labels
4. Train the model
5. Deploy the model into the car

### Extract Frames from Video
A video file is split in individual frames using the `ffmpeg` command:
```
ffmpeg -i <INPUT-FILE> -qscale:v 2 img%05d.jpg
```
> the `qscale:v 2` option is to obtain JPEG images with the best possible quality

Each frame is a JPEG image file with a name containing sequential numbers in order of time.

Example:
```
# create a directory to contain the frames using the timestamp in the filename
mkdir images-1482024251
# change directory and run the ffmpeg command
cd images-1482024251
ffmpeg -i ../74DA388EAC61-1482024251.flv -qscale:v 2  img%05d.jpg
```
> At 30 fps acquisition rate, every minute of video results in 1800 JPEG images.

### Extract Telemetry from Frames

Steering and throttle are embedded in each frame in a 40x10 pixel box in the bottom left of the image as three-digit numbers, using a fixed-size font. Each digit is a 7x5 image.

A CNN coded in `Keras/Tensorflow` has been trained to recognize each digit in the telemetry box. The code for the CNN, togheter with the `digit_cnn.json` Keras model and the `digit_cnn.h5` weights, is in the `ConvNet/DigitsNet` directory. About 4,000 images have been used for the training. Training images and their labels are in the files `testimages.npy` and `testlabels.npy`

The `ConvNet/extract_telemetry.py` Python script is used to extract telemetry from all the still images in a directory, and creating the `labels.csv` file in the same directory:
```
extract_telemetry.py <IMAGE-DIR>
```
Example:
```
cd Autonomia/ConvNet
extract_telemetry.py /home/oem/images-1482024251
```
At completion, the `images-1482024251/labels.csv` is created and it contains labels as follows:
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

### Prepare NumPy Arrays

The images need to be converted in YUV and together with the labels, stored as NumPy Array, and ready as input to train the model. This way several models or the same mode with different hyper-parameters can be trained without repeating this step.

In the current implementation, once uncompressed and converted, only the Y-plane of the images is used for training the CNN. Also, labels are converted into one-hot vectors, for proper use as categorical variables by the model.

The `ConvNet/prepare_data.py` Python script is used to convert images and labels into NumPy variables, and to create the `X_yuv_gray.npy`, `y1_steering.npy`, and `y2_throttle.npy` NumPy array files:
```
prepare_data.py <IMAGE-DIR>
```
Example:
```
cd Autonomia/ConvNet
prepare_data.py /home/oem/images-1482024251
```
> This script can be run in an interactive way, showing the image at each step, by modifying the `interactive` boolean variable in the script.

### Train the Model

The CNN model in `Keras` is defined in the `train.py` Python script. The hyperparameters are (mostly) in `config.py`, with some exceptions that are hard-coded (learning rate, optimization function).

```
train.py <IMAGE-DIR>
```
Example:
```
cd Autonomia/ConvNet
train.py /home/oem/images-1482024251
```
Once the training is completed, the model is saved as JSON model and the weights in the HDF5 binary format, in the files `autonomia_cnn.json` and `autonomia_cnn.h5`.

>The main application loads the model and weights from the name indicated in the `config.json` file.

