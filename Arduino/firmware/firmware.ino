/*
 * Firmware for Autonomia low-level Arduino controller.
 * 
 * The low-level controller intercepts from the receiver the values for throttle and servo and sends them to the host.
 * From the host receives from the PWM settings for throttle and servo.
 * 
 * Copyright (C) 2016 Visible Energy, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <Servo.h>
#include <EnableInterrupt.h>

// Pins mapping
const int THROTTLE_PIN = 4;
const int STEERING_PIN = 5;
const int MOTOR_PIN = 10;
const int SERVO_PIN= 11;
const int LED_PIN = 13; // the on-board L LED

// Globals
bool debug = false;
volatile uint8_t updateFlagsShared;
uint8_t updateFlags;
const int THROTTLE_FLAG = 1;
const int STEERING_FLAG = 2;
 
uint32_t throttleStart;
uint32_t steeringStart;
volatile uint16_t throttleInShared;
volatile uint16_t steeringInShared;

// Cctual throttle range is [980, 1980  === [253.176]
uint16_t throttleIn = 1500;
// Cctual steering range is [1000, 1984] === [0. 177] -- TODO: do we need calibration?
uint16_t steeringIn = 1500;

// Motor limits  -- TODO: not used
const int MOTOR_MAX = 120;
const int MOTOR_MIN = 40;
const int MOTOR_NEUTRAL = 90;

// Steering limits -- TODO: not used
const int D_THETA_MAX = 30;
const int THETA_CENTER = 90;
const int THETA_MAX = THETA_CENTER + D_THETA_MAX;
const int THETA_MIN = THETA_CENTER - D_THETA_MAX;
    
// Interfaces to motor and steering actuators
Servo motor;
Servo steering;

String inputLine = "";
bool rawOutput = false;
bool isConnected = true;
unsigned long lastHeartbeat = 0;

void initActuators() {
  motor.attach(MOTOR_PIN);
  steering.attach(SERVO_PIN);
}

void armActuators() {
  motor.write(MOTOR_NEUTRAL);
  steering.write(THETA_CENTER);
  delay(1000);
}

// RC steering input interrupt service routine
void steeringISR() {
  if(digitalRead(STEERING_PIN) == HIGH) {
    steeringStart = micros();
  } else {
    steeringInShared = (uint16_t)(micros() - steeringStart);
    updateFlagsShared |= STEERING_FLAG;
  }
}

// RC throttle input interrupt service routine
void throttleISR() {
  if(digitalRead(THROTTLE_PIN) == HIGH) {
    // rising edge of the signal pulse, start timing
    throttleStart = micros();
  } else {
    // falling edge, calculate duration of throttle pulse
    throttleInShared = (uint16_t)(micros() - throttleStart);
    // set the throttle flag to indicate that a new signal has been received
    updateFlagsShared |= THROTTLE_FLAG;
  }
}

void initRCInput() {
  pinMode(THROTTLE_PIN, INPUT_PULLUP);
  pinMode(STEERING_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); 
  enableInterrupt(THROTTLE_PIN, throttleISR, CHANGE);
  enableInterrupt(STEERING_PIN, steeringISR, CHANGE);
}

// Handle inputs from RC
void readAndCopyInputs() {
  // check global update flags to see if any channels have a new signal
  if (updateFlagsShared) {
    noInterrupts();
    // make local copies
    updateFlags = updateFlagsShared;
    if(updateFlags & THROTTLE_FLAG) {
      throttleIn = throttleInShared;
    }
    if(updateFlags & STEERING_FLAG) {
      steeringIn = steeringInShared;
    }
    // clear shared update flags and enable interrupts
    updateFlagsShared = 0;
    interrupts();
  }
}

// Scale RC pulses from 1000 - 2000 microseconds to 0 - 180 PWM angles
uint8_t microseconds2PWM(uint16_t microseconds) {
  if (microseconds < 1000)
    microseconds = 1000;
  
  microseconds -= 1000;
  uint16_t pwm = int(microseconds *.180 + .5);
  
  if (pwm < 0)
    pwm = 0;
  if (pwm > 180)
    pwm = 180;
  return static_cast<uint8_t>(pwm);
}

/*
  Parse inputLine received from host
  
  M [val]   -- set motor to val 
  S [val]   -- set servo to val
  H         -- heartbeat
  R         -- serial output raw
  V         -- serial output in range [0, 180] (default)

  Change parameters of servo and motor values accordingly
  return true if steering or throttle values have changed
*/ 
bool cmdParse(uint8_t *rc_outputs_steering, uint8_t *rc_outputs_throttle) {
  String val;
  if (inputLine.length() < 1)
    return false;
    
  int commandCode = inputLine[0];
  bool ret = false;
  
  Serial.println(inputLine);
  switch (commandCode) {
    case 'R':
      rawOutput = true;
      break;
    case 'V':
      rawOutput = false;
      break;
    case 'M':
      val = inputLine.substring(1);
      *rc_outputs_throttle = constrain(val.toInt(), 0, 180);
      ret = true;
      break;
    case 'S':
      val = inputLine.substring(1);
      *rc_outputs_steering = constrain(val.toInt(), 0, 180);
      ret = true;
      break;      
  }

  // treat each command as a heartbeat
  lastHeartbeat = millis();
  return ret;
}

void setup() {
  inputLine.reserve(128);
  
  initRCInput();
  initActuators();

  armActuators();
  Serial.begin(38400); //57600); // 38400);
}  

void loop() {
  static uint8_t rc_outputs_steering = THETA_CENTER;
  static uint8_t rc_outputs_throttle = MOTOR_NEUTRAL;   
  static unsigned long dt;
  static unsigned long t0;
  static uint8_t last_steeringIn;
  static uint8_t last_throttleIn;
  unsigned long now;
  uint8_t rc_inputs_steering;
  uint8_t rc_inputs_throttle;
     
  // check for connected flag
  if (!isConnected)
    return;
  
  // compute time elapsed from last loop
  now = millis();
  dt = now - t0;

  // handle inputs from radio receiver every 50 msec
  if (dt > 50) {
    readAndCopyInputs();

    // RC inputs scaled to [0, 180] range
    rc_inputs_throttle = microseconds2PWM(throttleIn);
    rc_inputs_steering = microseconds2PWM(steeringIn);

    // send readings to the host only when changed
    if ((0 < abs(last_throttleIn - throttleIn)) || (0< abs(last_steeringIn - steeringIn))) {
      // send readings to the host
      if (rawOutput) {
        Serial.print(throttleIn);
        Serial.print(" ");
        Serial.println(steeringIn);       
      } else {
        Serial.print(rc_inputs_throttle);
        Serial.print("  "); 
        Serial.println(rc_inputs_steering);
      }
    }
    t0 = millis();
  }

  // handle input from host
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n') {
      if (cmdParse(&rc_outputs_steering, &rc_outputs_throttle)) {
        if (debug) {
          Serial.print("****");       
          Serial.print(rc_outputs_steering);       
          Serial.println(rc_outputs_throttle);       
        }
        // output values have changed
        motor.write(rc_outputs_throttle);
        delay(15);
        steering.write(rc_outputs_steering);
        delay(15);
      }
      inputLine = "";
     } else
       inputLine += ch;
  }

/* DEBUG
  // check connection to host and stop the car if the heartbeat has not been received for a second
  if (lastHeartbeat + 1000 < now) {

    rc_outputs_steering = THETA_CENTER;
    rc_outputs_throttle = MOTOR_NEUTRAL;     
    motor.write(rc_outputs_throttle);
    steering.write(rc_outputs_steering);
    isConnected = false;
  }
*/
} // loop

