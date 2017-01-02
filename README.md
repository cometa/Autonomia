# Autonomia
A project for a cloud connected, autonomous RC 1/10 scale electric car to participate in the [DIY Robocars self-racing-cars events.](https://www.meetup.com/Self-Racing-Cars/) The project makes extensive use of [Vederly](http://www.cometa.io), a video and device management cloud platform for mobility applications developed by Visible Energy Inc.

The car controller software relies on a convolutional neural network, trained end-to-end using video from the camera, to drive itself predicting steering and throttle values from a live image.

**Video from the first test run:**

[![First Test Run](https://img.youtube.com/vi/f2dknc7g4Zc/0.jpg)](https://www.youtube.com/watch?v=f2dknc7g4Zc)
## Running Modes
The Autonomia vehicle has three running modes:

1. Controlled by the radio remote control (RC)
2. Controlled remotely through the Remote JSON/RPC API and the Vederly cloud servers
3. Autonomously, driven by the predictions of a convolutional neural network trained end-to-end

In any running mode, the vehicle is connected to the Vederly cloud servers using the [Cometa API](http://www.cometa.io/cometa-api.html). It is also remotely managed and responding to the commands in the JSON/RPC Remote API, as well streaming telemetry and live video from the on-board camera to the Vederly cloud managemengt platform.

## Hardware
The hardware added to a commercial 1/10 scale RC car, such as the [Traxxas Stampede](https://traxxas.com/products/models/electric/36054-1stampede?t=details) used in the first model consists of:

1. [Raspberry PI 3 model B](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/) with a 8GB SD card
2. [Arduino Nano](https://www.arduino.cc/en/Main/arduinoBoardNano)
3. [Logitech C920 camera](http://www.logitech.com/en-us/product/hd-pro-webcam-c920)
4. DC-DC step-down converter for automotive

Optional:

1. [Ublox GPS with USB interface](http://www.hardkernel.com/main/products/prdt_info.php?g_code=G142502154078) (needed if used outdoor)
2. [Adafruit 9 DFO Inertial Measurements Unit RPI shield](https://www.adafruit.com/products/2472)
3. [Maxbotix Ultrasonic Rangefinder](https://www.adafruit.com/products/172)

No changes have been done to the car chassis. The Arduino Nano is mounted on a protoboard and anchored with one of the screws used for the RC receiver. The camera is tight with a strap tiedown to the roof. The RPI is inside an enclosure and attached to the bottom of the chassis with a strap tiedown.

Power is supplied by the NiHM 3000 mAhr standard battery or a Lipo 5800 mAhr battery with a Y cable to power the RPI through an automotive DC-DC step-down power supply. The Arduino, camera and GPS are powered from their USB connections to the RPI.

The Arduino Nano receives the throttle and steering inputs from the radio receiver, and controls the inputs to the car motor ESC and steering servo. It also interfaces with the RPI to receive steering and throttle values as well as communicate to the RPI the readings from the radio controller. There is no direct connection between the radio receiver and the vehicle's servos.

## Software

The main application in `Python` consists of:

1. an implementation of a JSON/RPC remote API to control the vehicle from the Vederly cloud
2. a main car controller loop to operate the car motor and steering servos (through the Arduino interface)
3. a neural network model in `Keras`, trained end-to-end to predict steering and throttle from images

In parallel to the main application, an `ffmpeg` streamer is sending video to the Vederly cloud for live viewing inside a ground control station (GCS) application, and to store it for CNN training or driving evaluation purposes.

The CNN training video and telemetry data are acquired with the car controlled manually with the radio RC, and the `ffmpeg` streamer running in training mode, which allows for embedding the current steering and throttle values in the bottom left corner of the video itself. Steering and throttle values are then extracted frame per frame, as part of the data preparation and model training pipeline.

At any time together with live video, telemetry data are also sent at a selectable rate, to the Vederly cloud for use live in the GCS and to store for offline viewing.

The trained Keras model is loaded at runtime and is evaluated in about 40 milliseconds or less, depending on the number of nodes in the network. The model is evaluating steering and throttle values from a raw YUV 4:2:2 encoded frame, acquired by the streamer and stored in a shared RAM filesystem. The evaluated steering and throttle are passed to the Arduino controller to set the proper values for the motor and steering servo. In the current implementation, no loopback control mechanism is in place.

## Performance

The car runs autonomously very smoothly, with the main application in Python running on the Raspberry PI making steering and throttle predictions at the video acquisition rate of 30 frames per second. The main application does not perform any video processing, with the raw video acquired and resized by the streamer running in parallel and sharing data through a RAM filesystem.

Also, the `ffmpeg` streamer has been built to take advantage of the RPI GPU, which leaves most of the CPUs available to perform the model evaluation.

Since the CNN training is happening in the cloud, an inexpensive Raspberry PI and a small Arduino Nano is all the computing power needed on board the vehicle. The camera used is also encoding video in H.264 in its own hardware, requiring re-encoding by the RPI only while training to embed steering and throttle data in the video.

## Documentation

* [Remote cloud API](../master/docs/remote-api.md) 
* [CNN training pipeline](../master/ConvNet/README.md)
* [Arduino controller](../master/Arduino/README.md)
* [Dependencies](../master/docs/dependencies.md)

## Credits

While we are using a radically different approach and a minimalistic hardware, credit is given to Otavio Good and the [Carputer](https://github.com/otaviogood/carputer) team for showing feasibility of using a CNN for 1/10 scale cars autonomous driving, and for providing very useful insights in relevant design choices.
