#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import print_function
import serial
import time
import threading


class LidarProcessor:
    def __init__(self, port="/dev/ttyACM0", baudrate=460800,
                 distance_min=100, distance_max=5000,
                 min_consecutive_points=3,
                 region1=(300, 360), region2=(0, 60)):
        """
        初始化雷达处理器

        参数:
            port: 串口号 (默认: "/dev/ttyACM0")
            baudrate: 波特率 (默认: 460800)
            distance_min: 最小有效距离(mm) (默认: 100)
            distance_max: 最大有效距离(mm) (默认: 5000)
            min_consecutive_points: 最小连续点数要求 (默认: 3)
            region1: 第一个区域的角度范围 (默认: 300-360度)
            region2: 第二个区域的角度范围 (默认: 0-60度)
        """
        # 存储参数
        self.port = port
        self.baudrate = baudrate
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.min_consecutive_points = min_consecutive_points

        # 配置监测区域
        self.regions = {
            'region1': {'min': region1[0], 'max': region1[1]},
            'region2': {'min': region2[0], 'max': region2[1]}
        }

        # 串口连接
        self.ser = None
        self.running = False

        # 数据缓冲区
        self.latest_flags = "00"  # 默认值
        self.lock = threading.Lock()

    def connect(self):
        """连接雷达串口"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=5)
            print(f"已连接到雷达端口 {self.port}")
            return True
        except Exception as e:
            print(f"连接雷达失败: {e}")
            return False

    def disconnect(self):
        """断开雷达连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("雷达连接已关闭")

    def start(self):
        """启动雷达数据采集线程"""
        if not self.connect():
            return False

        # 打印配置信息
        print("\n雷达监测配置:")
        print(f"串口: {self.port} @ {self.baudrate}bps")
        print(f"有效距离范围: {self.distance_min}-{self.distance_max}mm")
        print(f"最小连续点数: {self.min_consecutive_points}")
        print(f"监测区域1: {self.regions['region1']['min']}-{self.regions['region1']['max']}度")
        print(f"监测区域2: {self.regions['region2']['min']}-{self.regions['region2']['max']}度")
        print("\n开始采集数据... (调用stop()停止)")

        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        return True

    def stop(self):
        """停止雷达数据采集"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.disconnect()
        print("雷达数据采集已停止")

    def get_flags(self):
        """
        获取最新的标志位状态
        返回: 两位字符串，如"01"表示区域1无物体，区域2有物体
        """
        with self.lock:
            return self.latest_flags

    def _parse_data(self, data):
        """解析雷达数据帧"""
        if len(data) < 105:
            return None

        start_angle = (data[2] * 256 + data[3]) / 100.0
        end_angle = (data[102] * 256 + data[103]) / 100.0

        region_flags = {
            'region1': {'count': 0, 'flag': 0},
            'region2': {'count': 0, 'flag': 0}
        }

        for i in range(32):
            idx = 4 + i * 6
            if idx + 5 >= len(data):
                break

            dist1 = data[idx] * 256 + data[idx + 1]  # 主回波
            dist2 = data[idx + 3] * 256 + data[idx + 4]  # 次回波

            angle = (start_angle + (end_angle - start_angle) * i / 32) % 360
            int_angle = int(round(angle))

            # 获取有效距离
            distance = 0
            if dist1 > 0 and self.distance_min <= dist1 <= self.distance_max:
                distance = dist1
            elif dist2 > 0 and self.distance_min <= dist2 <= self.distance_max:
                distance = dist2

            # 检查各区域
            for region in ['region1', 'region2']:
                min_angle = self.regions[region]['min']
                max_angle = self.regions[region]['max']

                # 处理角度跨越0度的情况
                if min_angle < max_angle:
                    in_region = min_angle <= int_angle <= max_angle
                else:
                    in_region = int_angle >= min_angle or int_angle <= max_angle

                if in_region and distance > 0:
                    region_flags[region]['count'] += 1
                    if region_flags[region]['count'] >= self.min_consecutive_points:
                        region_flags[region]['flag'] = 1

        return f"{region_flags['region1']['flag']}{region_flags['region2']['flag']}"

    def _run(self):
        """数据采集线程主循环"""
        try:
            while self.running:
                try:
                    data = self.ser.read(1)
                    if data[0] == 0xA5:
                        data = self.ser.read(1)
                        if data[0] == 0x5A:
                            data = self.ser.read(1)
                            if data[0] == 0x6C:
                                data = self.ser.read(105)
                                if len(data) == 105:
                                    flags = self._parse_data(data)
                                    if flags:
                                        with self.lock:
                                            self.latest_flags = flags
                except Exception as e:
                    continue
        finally:
            self.disconnect()