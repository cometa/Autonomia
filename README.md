![alt tag](https://img.shields.io/badge/python-2.7-blue.svg)

# Autonomia
A project for a cloud connected, autonomous RC 1/10 scale electric car to participate in the [DIY Robocars self-racing-cars events.](https://www.meetup.com/Self-Racing-Cars/) For video and telemetry data collection and the remote control API, the project relies on cloud API [Autonomia](http://www.autonomia.io), a video and device management cloud platform and API for mobility applications developed by `Visible Energy Inc. dba Autonomia`

The car autopilot car software running on the Raspberry PI on-board the car, is based on a convolutional neural network, trained end-to-end using video from the camera and it predicts steering and throttle values from a live image about 20 times per second on the RPI.

**Video from the first test run (Oakland warehouse):**

[![First Test Run](https://img.youtube.com/vi/3SsrNfRHWoU/0.jpg)](https://youtu.be/3SsrNfRHWoU)
## Running Modes
The Autonomia vehicle has three running modes:

1. Controlled by the radio remote control (RC)
2. Controlled remotely through the Remote JSON/RPC API and the Autonomia cloud servers
3. Autonomously, driven by the predictions of a convolutional neural network trained end-to-end

In any running mode, the vehicle is connected to the Autonomia cloud servers using the [Cometa API](http://www.cometa.io/cometa-api.html). It is also remotely managed and responding to the commands in the JSON/RPC Remote API, as well streaming telemetry and live video from the on-board camera to the Autonomia cloud managemengt platform.

## Hardware
The hardware added to a commercial 1/10 scale RC car, such as our first car "Gina", a [Traxxas Stampede](https://traxxas.com/products/models/electric/36054-1stampede?t=details), consists of:

1. [Raspberry PI 3 model B](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/) with a 8GB SD card
2. [Arduino Nano](https://www.arduino.cc/en/Main/arduinoBoardNano)
3. [Logitech C920 camera](http://www.logitech.com/en-us/product/hd-pro-webcam-c920)
4. DC-DC step-down converter for automotive

Our second car "Lola" is based on the 1/8 scale [Thunder Tiger MT4 G3 4WD Monster Truck](http://www.thundertiger.com/products-detail.php?id=10&lang=en)

Optional equipmment:

1. [Ublox GPS with USB interface](http://www.hardkernel.com/main/products/prdt_info.php?g_code=G142502154078) (needed if used outdoor)
2. [Adafruit 9 DFO Inertial Measurements Unit RPI shield](https://www.adafruit.com/products/2472)
3. [Maxbotix Ultrasonic Rangefinder](https://www.adafruit.com/products/172)

No changes have been made to the car chassis. The Arduino Nano is mounted on a protoboard and anchored with one of the screws used for the RC receiver. The camera is tight with a strap tiedown to the roof. The RPI is inside an enclosure and attached to the bottom of the chassis with a strap tiedown.

Power is supplied by the NiHM 3000 mAhr standard battery or a Lipo 5000 mAhr battery with a Y cable to power the RPI through an automotive DC-DC step-down power supply. The Arduino, camera and GPS are powered from their USB connections to the RPI.

The Arduino Nano receives the throttle and steering inputs from the radio receiver, and controls the inputs to the car motor ESC and steering servo. It also interfaces with the RPI to receive steering and throttle values as well as communicate to the RPI the readings from the radio controller. There is no direct connection between the radio receiver and the vehicle's servos.

## Software

The main application in `Python` consists of:

1. an implementation of a `JSON/RPC` remote API to control the vehicle from the Autonomia cloud
2. a main car controller loop to operate the car motor and steering servos (through the Arduino interface)
3. a neural network model in `Keras`, trained end-to-end to predict steering and throttle from images

In parallel to the main application, an `ffmpeg` streamer is sending video to the Autonomia cloud for live viewing inside a ground control station (GCS) application, and to store it for CNN training or driving evaluation purposes.

The CNN training video and telemetry data are acquired with the car controlled manually with the radio RC, and the `ffmpeg` streamer running in training mode, which allows for embedding the current steering and throttle values in the bottom left corner of the video itself. Steering and throttle values are then extracted frame per frame, as part of the data preparation and model training pipeline.

At any time together with live video, telemetry data are also sent at a selectable rate, to the Autonomia cloud for use live in the GCS and to store for offline viewing.

The trained `Keras` model (`Tensorflow` back-end) is loaded at runtime and is evaluated in about 40 milliseconds or less, depending on the number of nodes in the network. The model is evaluating steering and throttle values from a `YUV 4:2:2` encoded frame, acquired by the streamer at 30 fps. The evaluated steering and throttle are passed to the Arduino controller to set the proper values for the motor and steering servos. In the current implementation, no loopback control mechanism is in place.

## Performance

The car runs autonomously very smoothly, with the main application in Python running on the Raspberry PI making steering and throttle predictions at the video acquisition rate of 30 frames per second. The main application does not perform any video processing, with the raw video acquired and resized by the streamer running in parallel and sharing data through a video pipeline.

Also, the `ffmpeg` streamer has been built to take advantage of the RPI GPU, which leaves most of the CPUs available to perform the model evaluation.

Since the CNN training is happening in the cloud, an inexpensive Raspberry PI and a small Arduino Nano is all the computing power needed on board the vehicle. The camera used is also encoding video in H.264 in its own hardware, requiring re-encoding by the RPI only while training to embed steering and throttle data in the video.

## Documentation

* [Remote cloud API](../master/docs/remote-api.md) 
* [CNN training pipeline](../master/ConvNet/README.md)
* [Arduino controller](../master/Arduino/README.md)
* [Dependencies](../master/docs/dependencies.md)

## Cloud Server and API
The application uses the Autonomia,io a video and device management cloud platform for mobility applications, including a two-way message broker for device-to-cloud and cloud-to-device secure communication. It runs on a server in the cloud and it uses HTTPS and secure WebSockets for efficient remote interaction of applications and vehicles.

The main application manages the connection to the Autonomia server as defined in the `config.json` parameters file, streams video using RTMP to the server, and exposes methods for JSON-RPC remote procedure calls to the vehicle. 

If you are interested in receiving beta tester credentials and access to a Autonomia cloud server for testing the Autonomia software or the cloud API send an email to info@autonomia.io
>Teams participating to the `DYI Robocars` events can obtain free use of the Autonomia server and storage (within limits).

To use with Postman:
* [Swagger cloud API definition file](../master/docs/Autonomia.postman_collection.json)
* [Swagger cloud API environment file](../master/docs/Autonomia.postman_environment.json)

## Credits

While we are using a radically different approach and a minimalistic hardware, credit is given to Otavio Good and the [Carputer](https://github.com/otaviogood/carputer) team for showing feasibility of using a CNN for 1/10 scale cars autonomous driving, and for providing very useful insights in relevant design choices.

We also credit DYI Robocars' Chris Anderson for organizing and driving the self-driving cars open source movement.
