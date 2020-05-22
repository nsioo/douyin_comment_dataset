import time
import os




comment = 'adb shell input tap 990 1440'
close = 'adb shell input tap 577 315'
start = 'adb shell input tap 330 500'
slide = 'adb shell input swipe 550 500 550 1000 100'

# os.system(adb1)
# time.sleep()

while True:
    os.system(comment)
    time.sleep(1.5)
    os.system(close)
    os.system(slide)
    time.sleep(1)








