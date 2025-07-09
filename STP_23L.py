#!/usr/bin/env python3
import serial
import time
import logging
import struct
from threading import Thread, Lock
from queue import Queue

# Serial port configuration
port = '/dev/ttyUSB0'
baudrate = 230400
timeout = 0.5
retry_interval = 0.1

# Shared data
distance_queue = Queue(maxsize=1)  # ä»ä¿çææ°æµéå¼
stop_thread = False
lock = Lock()

def parse_data(data):
    """è§£æLiDARæ°æ®åï¼STP-23Låå·ï¼"""
    if len(data) < 195 or data[0] != 0xAA or data[1] != 0xAA or data[2] != 0xAA or data[3] != 0xAA:
        logging.warning("æ ææ°æ®åï¼å¤´é¨ä¸å¹éææ°æ®ä¸å®æ´")
        return None

    if data[5] != 0x02:
        logging.warning(f"æå¤çå½ä»¤ç : {data[5]}")
        return None

    data_len = data[8] + (data[9] << 8)
    if data_len != 0xB8:
        logging.warning(f"æå¤çæ°æ®é¿åº¦: {data_len}")
        return None

    measurements = []
    packet_size = 15
    data_start = 10

    for i in range(data_start, data_start + 12 * packet_size, packet_size):
        if i + packet_size > len(data):
            break

        point_data = bytes(data[i:i + packet_size])
        try:
            distance, noise, peak, confidence, intg, reftof = struct.unpack('<HHIfBH', point_data)
            measurements.append((distance, noise, peak, confidence, intg, reftof))
        except struct.error as e:
            logging.error(f"è§£ææµéç¹å¤±è´¥: {e}")
            continue

    return measurements

def read_serial_data():
    """æç»­è¯»åä¸²å£æ°æ®"""
    global stop_thread
    while not stop_thread:
        try:
            with serial.Serial(port, baudrate, timeout=timeout) as ser:
                logging.info("ä¸²å£å·²æåæå¼")
                ser.reset_input_buffer()
                buffer = bytearray()

                while not stop_thread:
                    try:
                        data = ser.read(ser.in_waiting or 1)
                        if data:
                            buffer.extend(data)

                            while len(buffer) >= 4:
                                if buffer[0] == 0xAA and buffer[1] == 0xAA and buffer[2] == 0xAA and buffer[3] == 0xAA:
                                    if len(buffer) >= 195:
                                        packet = buffer[:195]
                                        buffer = buffer[195:]
                                        data = list(packet)
                                        measurements = parse_data(data)
                                        if measurements:
                                            with lock:
                                                # æ¸ç©ºæ§æ°æ®å¹¶å­å¥æ°å¼
                                                while not distance_queue.empty():
                                                    distance_queue.get_nowait()
                                                distance_queue.put(measurements[0][0])
                                                logging.debug(f"æ´æ°è·ç¦»å¼: {measurements[0][0]} mm")  # è°è¯æ¥å¿
                                    else:
                                        break  # ç­å¾æ´å¤æ°æ®
                                else:
                                    buffer.pop(0)  # ä¸¢å¼æ æå­è
                    except serial.SerialException as e:
                        logging.error(f"ä¸²å£è¯»åéè¯¯: {e}")
                        break
                    except Exception as e:
                        logging.error(f"æå¤éè¯¯: {e}")
                        break

        except serial.SerialException as e:
            logging.error(f"æ æ³æå¼ä¸²å£: {e}")
            if not stop_thread:
                time.sleep(retry_interval)
        except Exception as e:
            logging.error(f"æå¤éè¯¯: {e}")
            if not stop_thread:
                time.sleep(retry_interval)

def get_distance():
    """è·åææ°è·ç¦»æµéå¼ï¼éé»å¡ï¼"""
    with lock:
        if not distance_queue.empty():
            return distance_queue.get_nowait()  # ç´æ¥è¿åææ°å¼
    return 9999

def start_lidar_thread():
    """å¯å¨LiDARæ°æ®è¯»åçº¿ç¨"""
    thread = Thread(target=read_serial_data, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # è°è¯æ¨¡å¼è¾åºæ´å¤ä¿¡æ¯
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    lidar_thread = start_lidar_thread()

    try:
        while True:
            distance = get_distance()
            if distance is not None:
                print(f"\rå½åè·ç¦»: {distance} mm", end="", flush=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        stop_thread = True
        lidar_thread.join()
        print("\nç¨åºç»æ­¢")