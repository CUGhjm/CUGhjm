import smbus
import time

class BH1750:
    # 定义 BH1750 的 I2C 地址
    ADDR = 0x23  # 默认地址
    # 定义 BH1750 的命令
    POWER_DOWN = 0x00  # 无主动状态
    POWER_ON = 0x01  # 等待测量命令
    RESET = 0x07  # 重置设备，使用默认值
    ONE_TIME_HIGH_RES_MODE = 0x20  # 一次高分辨率测量

    def __init__(self, bus_num=0):
        """
        初始化 BH1750 传感器
        :param bus_num: I2C 总线编号，默认为 5
        """
        self.bus = smbus.SMBus(bus_num)
        self.addr = self.ADDR
        self.power_down()
        self.power_on()
        time.sleep(0.01)  # 等待设备初始化

    def power_down(self):
        """
        关闭 BH1750 传感器
        """
        self.bus.write_byte(self.addr, self.POWER_DOWN)

    def power_on(self):
        """
        打开 BH1750 传感器
        """
        self.bus.write_byte(self.addr, self.POWER_ON)

    def reset(self):
        """
        重置 BH1750 传感器
        """
        self.bus.write_byte(self.addr, self.RESET)
        time.sleep(0.01)  # 等待设备重置

    def read_light(self):
        """
        读取光照强度
        :return: 光照强度值（单位：lux）
        """
        self.bus.write_byte(self.addr, self.ONE_TIME_HIGH_RES_MODE)
        time.sleep(0.18)  # 等待测量完成
        data = self.bus.read_i2c_block_data(self.addr, 0x00, 2)
        result = (data[0] << 8) | data[1]
        return result

# if __name__ == "__main__":
#     try:
#         sensor = BH1750(bus_num=0)  # 使用 I2C3 总线
#         while True:
#             light_level = sensor.read_light()
#             print(f"光照强度: {light_level} lux")
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("程序已退出")

# sensor = bh1750.BH1750(bus_num=0)
# light_level = sensor.read_light()