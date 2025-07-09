# -*- encoding=utf-8 -*-
import serial
import time
import logging


class PhoneModule:
    def __init__(self, serial_port="/dev/ttyUSB2", baudrate=115200):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser = serial.Serial()
        self.is_open = False
        self.call_status = False
        logging.basicConfig(level=logging.INFO)

    def open_serial(self):
        """打开串口连接"""
        try:
            if not self.is_open:
                self.ser.port = self.serial_port
                self.ser.baudrate = self.baudrate
                self.ser.bytesize = 8
                self.ser.stopbits = 1
                self.ser.parity = "N"
                self.ser.open()
                if self.ser.isOpen():
                    self.is_open = True
                    logging.info("串口打开成功")
                    return True
            return False
        except Exception as e:
            logging.error(f"打开串口时出错: {e}")
            return False

    def close_serial(self):
        """关闭串口连接"""
        try:
            if self.is_open:
                self.ser.close()
                self.is_open = False
                logging.info("串口已关闭")
                return True
            return False
        except Exception as e:
            logging.error(f"关闭串口时出错: {e}")
            return False

    def send_at_command(self, command, expected_response="OK", timeout=1):
        """发送AT命令并检查响应"""
        try:
            if not self.is_open:
                logging.error("串口未打开")
                return False

            self.ser.write((command + '\r\n').encode())
            time.sleep(timeout)

            if self.ser.inWaiting():
                time.sleep(0.1)
                response = self.ser.read(self.ser.inWaiting()).decode()
                if expected_response in response:
                    logging.info(f"命令成功: {command} -> 响应: {response.strip()}")
                    return True
                else:
                    logging.error(f"命令失败: {command} -> 预期: {expected_response}, 实际: {response.strip()}")
                    return False
            else:
                logging.error(f"命令无响应: {command}")
                return False
        except Exception as e:
            logging.error(f"发送AT命令时出错: {e}")
            return False

    def make_call(self, phone_number, call_duration=3):
        """拨打电话并自动挂断"""
        try:
            if not self.open_serial():
                logging.error("无法打开串口连接")
                return False

            # 检查模块是否就绪
            if not self.send_at_command("AT"):
                logging.error("模块未响应AT命令")
                return False

            # 拨打电话
            dial_command = f"ATD{phone_number};"
            if not self.send_at_command(dial_command):
                logging.error("拨号失败")
                return False

            self.call_status = True
            logging.info(f"正在拨打 {phone_number}...")

            # 等待指定时间后挂断
            time.sleep(call_duration)

            # 挂断电话
            if not self.send_at_command("ATH"):
                logging.error("挂断电话失败")
                return False

            self.call_status = False
            logging.info("电话已挂断")

            # 关闭串口
            self.close_serial()
            return True

        except Exception as e:
            logging.error(f"拨打电话过程中出错: {e}")
            self.close_serial()
            return False