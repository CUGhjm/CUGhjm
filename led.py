#!/usr/bin/env python3
import sys
import signal
import Hobot.GPIO as GPIO
import time

# 信号处理
def signal_handler(signal, frame):
    sys.exit(0)

def led_on():
    signal.signal(signal.SIGINT, signal_handler)
    # 设置管脚编码模式为硬件编号 BOARD
    GPIO.setmode(GPIO.BOARD)
    # 定义使用的GPIO通道为36
    output_pin = 36  # BOARD 编码 36
    # 设置为输出模式，并且初始化为高电平
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.output(output_pin, GPIO.HIGH)


def led_off():
    signal.signal(signal.SIGINT, signal_handler)
    # 设置管脚编码模式为硬件编号 BOARD
    GPIO.setmode(GPIO.BOARD)
    # 定义使用的GPIO通道为36
    output_pin = 36  # BOARD 编码 36
    # 设置为输出模式，并且初始化为高电平
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.output(output_pin, GPIO.LOW)


