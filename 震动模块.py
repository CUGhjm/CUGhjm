#!/usr/bin/env python3

################################################################################
# Copyright (c) 2024,D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
import signal
import Hobot.GPIO as GPIO
import time


def signal_handler(signal, frame):
    sys.exit(0)


# 支持PWM的管脚: 32 and 33, 在使用PWM时，必须确保该管脚没有被其他功能占用
output_pin1 = 32  # 新增的32号引脚
output_pin2 = 33  # 原有的33号引脚

GPIO.setwarnings(False)


def setup_pwm():
    # Pin Setup:
    # Board pin-numbering scheme
    signal.signal(signal.SIGINT, signal_handler)
    GPIO.setmode(GPIO.BOARD)

    # 初始化两个PWM输出
    # 支持的频率范围：48KHz ~ 192MHz
    p1 = GPIO.PWM(output_pin1, 48000)  # 32号引脚PWM
    p2 = GPIO.PWM(output_pin2, 48000)  # 33号引脚PWM

    # 初始占空比 25%，先每0.25秒增加5%占空比，达到100%之后再每0.25秒减少5%占空比
    val = 75
    p1.ChangeDutyCycle(val)
    p2.ChangeDutyCycle(val)
    p1.start(val)
    p2.start(val)

    print("PWM running on pins 32 and 33. Press CTRL+C to exit.")
    try:
        for i in range(0, 10):
            time.sleep(0.02)
            val = 75
            p1.ChangeDutyCycle(val)
            p2.ChangeDutyCycle(val)
    finally:
        p1.stop()
        p2.stop()
        GPIO.cleanup()


