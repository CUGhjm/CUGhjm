# -*- encoding=utf-8 -*-
import serial
import time
from threading import Timer

# 全局变量
ser = serial.Serial()
rec_buff = b''

def send_at(command, back, timeout):
    global rec_buff
    ser.write((command + '\r\n').encode())
    time.sleep(timeout)
    if ser.inWaiting():
        time.sleep(0.1)
        rec_buff = ser.read(ser.inWaiting())
    if rec_buff != b'':
        if back not in rec_buff.decode():
            print(command + ' ERROR')
            print(command + ' back:\t' + rec_buff.decode())
            return 0
        else:
            print(rec_buff.decode())
            return 1
    else:
        print(command + ' no response')
        return 0

def open_serial(serial_port, baudrate=115200):
    global ser
    ser.port = serial_port
    ser.baudrate = baudrate
    ser.bytesize = 8
    ser.stopbits = 1
    ser.parity = "N"
    try:
        ser.open()
        if ser.isOpen():
            print("串口打开成功！")
            return True
        else:
            print("串口打开失败！")
            return False
    except Exception as e:
        print('异常：', e)
        return False

def close_serial():
    global ser
    try:
        if ser.isOpen():
            ser.close()
            if ser.isOpen():
                print("串口关闭失败！")
                return False
            else:
                print("串口关闭成功！")
                return True
        else:
            print("串口未打开！")
            return True
    except Exception as e:
        print('异常：', e)
        return False

def get_location():
    global rec_buff
    if send_at('AT+CLBS=1', '+CLBS:', 2):
        rec_buff1 = rec_buff.decode()
        # 修改解析逻辑，确保正确提取经纬度
        start_index = rec_buff1.find("+CLBS: 0,") + len("+CLBS: 0,")
        if start_index != -1:
            end_index = rec_buff1.find("\n", start_index)
            if end_index == -1:
                end_index = len(rec_buff1)
            location_data = rec_buff1[start_index:end_index].strip()
            if ',' in location_data:
                latitude, longitude = location_data.split(',', 1)
                print("经纬度：", latitude, longitude)
                return latitude.strip(), longitude.strip()
        print("无法解析经纬度")
        return None, None
    else:
        print("获取经纬度失败")
        return None, None

def send_sms(phone_number, message):
    global rec_buff
    if send_at('ATE1', 'OK', 1):
        if send_at('AT+CSQ', 'OK', 1):
            if send_at('AT+CGATT?', '+CGATT: 1', 1):
                if send_at('AT+CSCS="GSM"', 'OK', 1):
                    if send_at('AT+CMGF=1', 'OK', 1):
                        if send_at(f'AT+CMGS="{phone_number}"', '>', 1):
                            ser.write(message.encode())
                            if send_at(chr(26), 'OK', 5):
                                print("短信发送成功！")
                                return True
    print("短信发送失败")
    return False

def send_fall_detected_sms(phone_number, latitude, longitude):
    message = f"Fall detected. Current latitude and longitude are: {latitude}, {longitude}"
    return send_sms(phone_number, message)