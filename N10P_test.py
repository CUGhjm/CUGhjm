#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from N10P import LidarProcessor
import time


def main():
    # 初始化雷达处理器，可自定义两个区域的角度范围
    lidar = LidarProcessor(
        port="/dev/ttyACM0",      # 串口号
        baudrate=460800,          # 波特率
        distance_min=100,         # 最小检测距离(mm)
        distance_max=800,         # 最大检测距离(mm)
        min_consecutive_points=2, # 最小连续点数
        region1=(270, 360),       # 第一个区域角度范围
        region2=(0, 90)           # 第二个区域角度范围
    )

    # 启动雷达
    if not lidar.start():
        return

    try:
        # 示例：每0.5秒获取一次标志位
        while True:
            flags = lidar.get_flags()
            print(f"当前标志位: {flags} (区域1:{flags[0]}, 区域2:{flags[1]})")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n程序终止")
    finally:
        lidar.stop()


if __name__ == '__main__':
    main()