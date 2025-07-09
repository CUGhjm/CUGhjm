import time
from led import led_init, led_on, led_off

led_init()
while True:
    led_on()
    time.sleep(1)
    led_off()
    time.sleep(1)