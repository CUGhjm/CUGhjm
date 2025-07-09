#!/usr/bin/env python3
from warning_message import open_serial, close_serial, get_location, send_fall_detected_sms
import sys
import signal
import serial
import logging
import struct
import Hobot.GPIO as GPIO
import time
import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
import argparse
from sound import play_by_number  # 修改后的异步播放模块
from N10P import LidarProcessor
from zhendong import control_pwm
import deepseek
from MPU6050 import calibrate_sensor, detect_fall, start_fall_detection_thread, stop_fall_detection_thread
from a7670g import PhoneModule
from led import led_on, led_off
import bh1750


#环境光检测强度
LIGHT_LEVEL = 30

# 全局变量记录最后一次报警时间
last_alert_time_yolo = 0
ALERT_COOLDOWN_yolo = 5  # 冷却时间(秒)
last_alert_time_lidar = 0
ALERT_COOLDOWN_lidar = 5  # 冷却时间(秒)


# 新增：用于存储连续检测结果的队列
DETECTION_HISTORY = []
REQUIRED_CONSECUTIVE_DETECTIONS = 3
DETECTION_INTERVAL = 0.1  # 秒


def can_alert_yolo():
    """检查是否超过冷却期可以报警"""
    global last_alert_time_yolo
    return time.time() - last_alert_time_yolo > ALERT_COOLDOWN_yolo


def update_alert_time_yolo():
    """更新最后一次报警时间"""
    global last_alert_time_yolo
    last_alert_time_yolo = time.time()


def can_alert_lidar():
    """检查是否超过冷却期可以报警"""
    global last_alert_time_lidar
    return time.time() - last_alert_time_lidar > ALERT_COOLDOWN_lidar


def update_alert_time_lidar():
    """更新最后一次报警时间"""
    global last_alert_time_lidar
    last_alert_time_lidar = time.time()


def check_consecutive_detection(current_class):
    """检查是否连续检测到相同目标"""
    global DETECTION_HISTORY

    # 记录当前检测结果和时间戳
    DETECTION_HISTORY.append((current_class, time.time()))

    # 移除过期的检测记录（超过检测间隔的）
    current_time = time.time()
    DETECTION_HISTORY = [(cls, ts) for cls, ts in DETECTION_HISTORY
                         if current_time - ts <= DETECTION_INTERVAL * REQUIRED_CONSECUTIVE_DETECTIONS]

    # 检查是否有足够数量的连续检测
    if len(DETECTION_HISTORY) >= REQUIRED_CONSECUTIVE_DETECTIONS:
        # 检查所有最近的检测是否相同
        recent_detections = [cls for cls, ts in DETECTION_HISTORY[-REQUIRED_CONSECUTIVE_DETECTIONS:]]
        if all(d == current_class for d in recent_detections):
            DETECTION_HISTORY = []  # 重置检测历史
            return True
    return False


def WARNING_message():
    serial_port = "/dev/ttyUSB2"
    phone_number = "18334776376"  # 替换为实际手机号

    # 打开串口
    if open_serial(serial_port):
        # 获取经纬度
        latitude, longitude = get_location()
        if latitude and longitude:
            # 发送短信
            if send_fall_detected_sms(phone_number, latitude, longitude):
                print("短信发送成功！")
            else:
                print("短信发送失败")
        else:
            print("无法获取经纬度")
        # 关闭串口
        close_serial()
    else:
        print("无法打开串口")


def test_phone_call():
    # 配置参数
    serial_port = "/dev/ttyUSB2"  # 替换为你的串口号
    phone_number = "18334776376"  # 替换为要拨打的电话号码

    # 创建电话模块实例
    phone = PhoneModule(serial_port=serial_port)

    # 拨打电话并自动挂断
    success = phone.make_call(phone_number, call_duration=3)

    if success:
        print("电话拨打成功")
    else:
        print("电话拨打失败")


# 信号处理
def signal_handler(signal, frame):
    sys.exit(0)


# 激光雷达初始化
lidar = LidarProcessor(
    port="/dev/ttyACM0",  # 串口号
    baudrate=460800,  # 波特率
    distance_min=100,  # 最小检测距离(mm)
    distance_max=101,  # 最大检测距离(mm)
    min_consecutive_points=3,  # 最小连续点数
    region1=(270, 360),  # 第一个区域角度范围
    region2=(0, 90)  # 第二个区域角度范围
)
# GPIO设置
input_pin_key = 37  # BOARD 编码 37
GPIO.setwarnings(False)
# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO_TEST")


def map_string_to_number(input_str):
    mapping = {
        'green': '3',
        'red': '4',
        'cross': '5',
        'no_entry': '6',
    }
    return mapping.get(input_str, 'Unknown')




def main():
    time.sleep(1)
    flag_ledd = 0

    while True:
        # 等待按钮按下开始检测
        while True:
            time.sleep(0.1)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(input_pin_key, GPIO.IN)
            value = GPIO.input(input_pin_key)
            if value == GPIO.LOW:
                play_by_number('1')  # 异步播放开始检测提示音
                control_pwm()  # 振动0.5秒
                break

        # 初始化检测模型
        DISPLAY, frame_count, model, start_time, cap = deepseek.detect_start(display=True)
        calibrate_sensor()  # 陀螺仪初始化
        start_fall_detection_thread()  # 启动MPU6050检测线程
        lidar.start()  # 启动激光雷达
        sensor = bh1750.BH1750(bus_num=0)
        last_light_check_time = 0

        # 主检测循环
        while True:
            if detect_fall():
                control_pwm()  # 振动0.2秒
                play_by_number('10')  # 异步播放对应语音
                test_phone_call()
                WARNING_message()
                update_alert_time_yolo()  # 更新报警时间
                update_alert_time_lidar()  # 更新报警时间
                deepseek.detect_last(DISPLAY, frame_count, start_time, cap)
                stop_fall_detection_thread()
                while True:
                    time.sleep(1)

            current_time_light = time.time()
            if current_time_light - last_light_check_time > 10:
                pre_flag_led = flag_ledd
                light_level = sensor.read_light()
                if  light_level > LIGHT_LEVEL:
                    flag_ledd = 0
                else:
                    flag_ledd = 1
                if flag_ledd == 1 and pre_flag_led == 0:
                    led_on()
                    play_by_number('11')
                elif flag_ledd == 1:
                    led_on()
                else:
                    led_off()

                pre_flag_led = flag_ledd
                light_level = sensor.read_light()
                if  light_level > LIGHT_LEVEL:
                    flag_ledd = 0
                else:
                    flag_ledd = 1
                if flag_ledd == 1 and pre_flag_led == 0:
                    led_on()
                    play_by_number('11')
                elif flag_ledd == 1:
                    led_on()
                else:
                    led_off()

                last_light_check_time = current_time_light

            # 执行检测
            class_name, MP4, ids_count = deepseek.detect_main(DISPLAY, frame_count, model, start_time, cap)
            # 处理检测结果
            if MP4 and ids_count != "0":
                coco_name1 = class_name
                logger.info(f"检测到目标1: {coco_name1}")
            else:
                coco_name1 = 'zxy'
            if not MP4:  # 视频流结束
                break

            if coco_name1 in ('green', 'red', 'cross', 'no_entry'):
                # 修改：只有当连续三次检测到相同目标时才触发
                if check_consecutive_detection(coco_name1) and can_alert_yolo():
                    control_pwm()  # 振动0.2秒
                    play_by_number(map_string_to_number(coco_name1))  # 异步播放对应语音
                    update_alert_time_yolo()  # 更新报警时间

            lidar_flags = lidar.get_flags()
            print(f"当前标志位: {lidar_flags} (区域1:{lidar_flags[0]}, 区域2:{lidar_flags[1]})")
            if lidar_flags != '00' and can_alert_lidar():  # 检查冷却期
                update_alert_time_lidar()  # 更新报警时间
                if lidar_flags == '10':
                    control_pwm()  # 振动0.5秒
                    play_by_number('7')  # 异步播放开始检测提示音
                elif lidar_flags == '01':
                    control_pwm()  # 振动0.5秒
                    play_by_number('8')  # 异步播放开始检测提示音
                elif lidar_flags == '11':
                    control_pwm()  # 振动0.5秒
                    play_by_number('9')  # 异步播放开始检测提示音


            # 检查按钮是否按下停止检测
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(input_pin_key, GPIO.IN)
            value = GPIO.input(input_pin_key)
            if value == GPIO.HIGH:
                control_pwm()  # 振动0.2秒
                play_by_number('2')  # 异步播放停止检测提示音
                led_off()
                deepseek.detect_last(DISPLAY, frame_count, start_time, cap)
                lidar.stop()
                stop_fall_detection_thread()
                sensor.power_down()
                break


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()