#!/usr/bin/env python

# Copyright (c) 2024，WuChao D-Robotics.
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

import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
from time import time
import argparse
import logging

import deepseek

logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO_TEST")

def main():
    # 初始化
    DISPLAY, frame_count, model, start_time, cap = deepseek.detect_start(display=True)

    try:
        while True:
            # 执行检测
            class_name, MP4, ids_count = deepseek.detect_main(DISPLAY, frame_count, model, start_time, cap)
            # 处理结果
            if MP4 and ids_count != "0":
                result = class_name
                logger.info(f"检测到目标: {result}")
            else:
                result = None
            if not MP4:  # 视频结束
                break
            # 检查是否按下q键退出
            if DISPLAY and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 释放资源
        deepseek.detect_last(DISPLAY, frame_count, start_time, cap)


if __name__ == "__main__":
    main()