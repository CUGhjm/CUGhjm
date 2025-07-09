from a7670g import PhoneModule
import time


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


if __name__ == "__main__":
    test_phone_call()