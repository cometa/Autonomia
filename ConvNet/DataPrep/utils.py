"""
  Cloud connected autonomous RC car.

  Copyright 2016 Visible Energy Inc. All Rights Reserved.
"""
__license__ = """
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def do_log_mapping_to_buckets(a):
    return int(round(math.copysign(math.log(abs(a) + 1, 2.0), a))) + 7


throttle_max_range_map = [
    [80,0], # if t <= 80 -> o=0 # Breaking:
    [82,1], # elif t <= 82 -> o=1
    [84,2], # elif t <= 84 -> o=2
    [86,3], # elif t <= 86 -> o=3
    [87,4], # elif t <= 87 -> o=4 # Breaking ^
    
    [96,5], # elif t <= 96 -> o=5 # Neutral

    [97,6], # elif t <= 97 -> o=6 # Forward:
    [98,7], # elif t <= 98 -> o=7
    [99,8], # elif t <= 99 -> o=8
    [100,9], # elif t <= 100 -> o=9
    
    [101,10], # elif t <= 101 -> o=10
    [102,11], # elif t <= 102 -> o=11
    [105,12], # elif t <= 105 -> o=12
    [107,13], # elif t <= 107 -> o=13
    [110,14]  # elif t <= 110 -> o=14
]

map_back = {5:90}

def to_throttle_buckets(t):
    t = int(float(t)+0.5) #nearest

    for max_in_bucket,bucket in throttle_max_range_map:
        if t <= max_in_bucket:
            return bucket
    return 14

def from_throttle_buckets(t):
    t = int(float(t)+0.5)
    for ibucket,(max_in_bucket,bucket) in enumerate(throttle_max_range_map):
        if t == bucket:
            if map_back.has_key(bucket):
                return map_back[bucket]

            return max_in_bucket

    return 100 # Never happens, defensively select a mild acceleration


def invert_log_bucket(a):
  # Reverse the function that buckets the steering for neural net output.
  # This is half in filemash.py and a bit in convnet02.py (maybe I should fix)
  # steers[-1] -= 90
  # log_steer = math.copysign( math.log(abs(steers[-1])+1, 2.0) , steers[-1]) # 0  -> 0, 1  -> 1, -1 -> -1, 2  -> 1.58, -2 -> -1.58, 3  -> 2
  # gtVal = gtArray[i] + 7
  steer = a - 7
  original = steer
  steer = abs(steer)
  steer = math.pow(2.0, steer)
  steer -= 1.0
  steer = math.copysign(steer, original)
  steer += 90.0
  steer = max(0, min(179, steer))
  return steer

