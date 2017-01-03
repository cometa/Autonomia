# Cloud API

Once the Autonomia application is started on the vehicle, it will connect to the Cometa middleware server, the message broker component of the Vederly Cloud API. The application uses as credentials the `app_key` application key specified in the `config.json` file. The connection is kept open permanently and used for full-duplex communication with the vehicle through the Cometa server. 

Since the connection is initiated from within a NAT or a firewall on stadard port 443, which is typically open for outgoing traffic, the vechicle becomes accessible from an application using the `Autonomia` Cloud API without exposing or even knowing its public IP address.

An application that intends to communicate with an Autonomia vehicle, sends `JSON-RPC` requests through the Cometa cloud server `send` method.

>To use the Cometa API and the `send` method, an `APPLICATION_KEY` and a `COMETA_SECRET` are needed as credentials for authentication. Users of the Cometa Robotics cloud service can create applications and manage their vehicles in the cloud as well as develop applications using the API provided by `cometa-dronekit`. 

>Public availability of Vederly Cloud API service is planned for 1Q 2017. Send an email to cometa@visiblenergy.com to request early access.

#### Cometa Authentication
An application is authenticated by including an Authorization HTTPS request header in every request.

| HTTP HEADER                    | DESCRIPTION                 | VALUE                         |
 -------------------------------|-------------------------|-------------
|  `Authorization`              | authorization token      | OAuth {`COMETA_SECRET`}         

Example:

`Authorization: OAuth b4f51e1e6125dca873e5`

### Send RPC Message 

Send a JSON-RPC message to a vehicle.

```
POST /v1/applications/{APPLICATION_KEY}/devices/{DEVICE_ID}/send
``` 
The message containing the JSON-RPC request is in the POST `body`.

| URL PARAMETER        | DESCRIPTION                      | TYPE                            |
 ------------------|----------------------------------|-------------------------------------
| `APPLICATION_KEY` | Cometa Application Key           | String |
| `DEVICE_ID` | Device Cometa Id           | String |

The `DEVICE_ID` is the vehicle ID provided by `cometa-dronekit` when a vehicle is connected to a Cometa server. **The default value is the vehicle's MAC address.**

The `Autonomia` vehicle agent always reply to a JSON-RPC message with a JSON response.

Example:
```
$ curl -X POST -H 'Authorization: OAuth a724dc4811d507688' -H 'Content-type: application/json' \
    -d '{"jsonrpc":"2.0", "method":"video_start","params":{"telem":true},"id":7}' \
    https://vederly.cometa.io/v1/applications/a94660d971eca2879/devices/e984060007/send

{"jsonrpc": "2.0", "result": {"success": true}, "id": 7}
```
### WebSockets Endpoint

The Cometa server exposes a `WebSockets` endpoint for each vehicle connected to it. A vehicle WebSocket can be opened only once. To obtain an endpoint, an application must request a new WebSockets `DEVICE_KEY` every time is needed using the following HTTPS method:

```
GET /applications/{APPLICATION_KEY}/devices/{DEVICE_ID}/websocket
```

 URL PARAMETER        | DESCRIPTION                      | TYPE                            |
 ------------------|----------------------------------|-------------------------------------
| `APPLICATION_KEY` | Cometa Application Key           | String |
| `DEVICE_ID` | Device Cometa Id           | String |

The method returns a `DEVICE_KEY` that is used to obtain the WebSocket vehicle's endpoint as follows:

`wss://{COMETA_HOST}:{PORT}/websockets/{DEVICE_ID}/{DEVICE_KEY}`


Example:
```
$ curl -H 'Authorization: OAuth a724dc4811d507688' -H 'Content-type: application/json' \
    https://vederly.cometa.io/v1/applications/a94660d971eca2879/devices/e984060007/websocket

{
    "device_id":"e984060007",
    "device_key":"dc670dae876ee4f919de9e777c9bd98a5e182cd8"
}
```
WebSocket endpoint (one-time use only):

    wss://vederly.cometa.io/v1/websockets/e984060007/dc670dae876ee4f919de9e777c9bd98a5e182cd8

Opening a device `WebSocket` would fail if the vehicle is not connected. Upon vehicle disconnection, the `WebSocket` is closed by the server after the inactivity timeout period. **Immediately after opening a WebSocket, and without any other request, an application starts receiving telemetry messages at expiration of every period of the duration indicated in the `config.json` file.**

>On an open WebSocket an application receives both telemetry messages without requesting them, and responses to JSON-RPC requests.

`WebSockets` are asynchronous, full-duplex channels to exchange messages directly between an application and a remote vehicle running the `Autonomia` application. A WebSocket `send()` is relaying the message to the vehicle the same way as an HTTPS `send`. From the vehicle's standpoint messages are received the same way regardless the method used by an application, that is using `WebSockets` method `send()`, or a Cometa HTTPS `send`. **On an open WebSocket an application receives both telemetry messages without requesting them, and responses to JSON-RPC requests.**

> Before sending a message to a WebSocket an application should always check its `readyState` attribute to check the WebSocket connection is open and ready to communicate (`readyState === WS_OPEN`).

### Connected Vehicles

Get a list of vehicle connected to the Cometa server. 
```
GET /v1/applications/{APPLICATION_KEY}/devices
``` 

The message containing the JSON-RPC request is in the POST `body`.

| URL PARAMETER        | DESCRIPTION                      | TYPE                            |
 ------------------|----------------------------------|-------------------------------------
| `APPLICATION_KEY` | Cometa Application Key           | String |

Example:
```
$ curl -H 'Authorization: OAuth b4f51e1e6125dcc873e9' -H 'Content-type: application/json' \
    http://vederly.cometa.io/v1/applications/a0353b75b8fa61889d19/devices

{
    "num_devices": 11, 
    "devices": [
        "cc79cf45f1b4", 
        "cc79cf45f1d4", 
        "cc79cf45f421", 
        "cc79cf45f1ba", 
        "cc79cf45f400", 
        "cc79cf45f401", 
        "cc79cf45f2ab", 
        "cc79cf45f307", 
        "cc79cf45f221", 
        "cc79cf45f314", 
        "cc79cf45f2d8"
    ]
}
```

### Cometa Vehicle Presence

Get vehicle connection state and statistics information from the Cometa server. 
```
GET /v1/applications/{APPLICATION_KEY}/devices/{DEVICE_ID}
``` 

The message containing the JSON-RPC request is in the POST `body`.

| URL PARAMETER        | DESCRIPTION                      | TYPE                            |
 ------------------|----------------------------------|-------------------------------------
| `APPLICATION_KEY` | Cometa Application Key           | String |
| `DEVICE_ID` | Device Cometa Id           | String |

Example:
```
$ curl -H 'Authorization: OAuth b4f51e1e6125dcc873e9' -H 'Content-type: application/json' \
    http://vederly.cometa.io/v1/applications/a0353b75b8fa61889d19/devices/e984060007

{
    "device_id": "e984060007", 
    "ip_address": "73.202.12.128:64471", 
    "heartbeat": "1478378638", 
    "info": "Autonomia", 
    "connected_at": "1478373655", 
    "latency": "45", 
    "websockets": "1", 
    "net": {
        "received": {
            "bytes": "4237", 
            "messages": "12"
        }, 
        "sent": {
            "bytes": "34789", 
            "messages": "781"
        }
    }
}
```
**Latency is in milliseconds and indicates the average time of a round-trip from the server to the vehicle (message/response). Latencies of less than 100 msec for a round-trip are typical for vehicles connected in the US.**
>Vehicle connections are maintained by a regular heartbeat message sent by `Autonomia` to the server (60 seconds period). The Cometa server times out and disconnects a vehicle 90 seconds after receiving the last heartbeat message. A disconnected vehicle may appear to be connected for up to an additional 90 seconds, if it disconnect abruptly without cleanly close its socket connection. 

## Methods

### Video Devices
`video_devices`

List available video devices (v4l).

Example:
```
$ curl -X POST -H 'Authorization: OAuth a724dc4811d507688' -H 'Content-type: application/json' \
    -d '{"jsonrpc":"2.0","method":"video_devices","params":{},"id":7}' \
    https://vederly.cometa.io/v1/applications/a94660d971eca2879/devices/e984060007/send
```

### Set Telemetry Period
`set_telemetry_period`

Set telemetry period in seconds.

### Get Configuration
`get_config`

Get the configuration object in the `config.json` file.

### Get Status
`get_status`

Get the vehicle current status.

### Set Mode
`set_mode`

Set the vehicle running mode.

| MODE                   | DESCRIPTION                 | NOTES                         |
 ------------------------|-------------------------|-------------
|  `TRAINING`            | control from RC radio controller      | embedded telemetry in video for CNN training         
|  `REMOTE`            | rempte control with Cloud API methods    | use `set_throttle` and `set_steering` for control         
|  `AUTO`            | autonomous mode      | predicting steering and throttle from CNN          

### Set Steering
`set_steering`

Set the steering value for a vehicle running in `REMOTE` mode. 

>Values are in the [0,180] range, with 90 for neutral.

### Set Throttle
`set_throttle`

Set the throttle value for a vehicle running in `REMOTE` mode. 

>Values are in the [0,180] range, with 90 for neutral.

### Start
`start`

Start the vehicle in `TRAINING` mode.

### Stop
`stop`

Stop the vehicle and change state to `IDLE`.

### Video Start
`video_start`

Start video streaming to the Vederly server using the `streamer` and `camera key` set in `config.json`. Set `"telem":true` to embed steering and throttle values in the image.

### Video Stop
`video_stop`

Stop video streaming.

