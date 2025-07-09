# -*- encoding=utf-8 -*-
from warning_message import open_serial, close_serial, get_location, send_fall_detected_sms

def main():
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

if __name__ == "__main__":
    main()