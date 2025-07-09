from smbus2 import SMBus
import math
import time
import threading
from queue import Queue

# 寄存器地址
PWR_MGMT_1 = 0x6B
ACCEL_XOUT = 0x3B
GYRO_XOUT = 0x43

# 初始化 I2C
bus = SMBus(5)  # i2cdetect 5
address = 0x68  # MPU6050

# 摔倒检测阈值
FALL_THRESHOLD = 1.5  # 加速度阈值（g）
ANGLE_THRESHOLD = 45  # 倾角阈值（度）

# 全局变量和队列用于线程间通信
fall_detection_queue = Queue(maxsize=1)
stop_event = threading.Event()


def read_word(reg):
    high = bus.read_byte_data(address, reg)
    low = bus.read_byte_data(address, reg + 1)
    return (high << 8) + low


def read_word_2c(reg):
    val = read_word(reg)
    return val if val < 0x8000 else val - 0x10000


def calibrate_sensor():
    bus.write_byte_data(address, PWR_MGMT_1, 0x00)  # 唤醒设备
    time.sleep(0.1)


def mpu6050_worker():
    """后台运行的MPU6050数据采集线程"""
    while not stop_event.is_set():
        try:
            # 读取加速度计数据
            accel_x = read_word_2c(ACCEL_XOUT) / 16384.0
            accel_y = read_word_2c(ACCEL_XOUT + 2) / 16384.0
            accel_z = read_word_2c(ACCEL_XOUT + 4) / 16384.0

            # 检测加速度是否超过阈值
            fall_detected = (abs(accel_x) > FALL_THRESHOLD or
                             abs(accel_y) > FALL_THRESHOLD or
                             abs(accel_z) > FALL_THRESHOLD)

            # 如果队列为空或者当前状态与之前不同，则更新队列
            if fall_detection_queue.empty() or fall_detection_queue.queue[0] != fall_detected:
                fall_detection_queue.put(fall_detected)

            time.sleep(0.05)  # 控制采样频率

        except Exception as e:
            print(f"MPU6050 worker error: {e}")
            time.sleep(0.1)


def start_fall_detection_thread():
    """启动摔倒检测线程"""
    global mpu_thread
    mpu_thread = threading.Thread(target=mpu6050_worker, daemon=True)
    mpu_thread.start()


def stop_fall_detection_thread():
    """停止摔倒检测线程"""
    stop_event.set()
    if mpu_thread.is_alive():
        mpu_thread.join()


def detect_fall():
    """
    检测是否摔倒
    :return: True 如果检测到摔倒，False 如果未检测到摔倒
    """
    if not fall_detection_queue.empty():
        return fall_detection_queue.get()
    return False