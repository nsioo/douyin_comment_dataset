import time
import os




#comment = 'adb -s 10.249.243.227:5555 shell input tap 990 1440'
#close = 'adb -s 10.249.243.227:5555 shell input tap 577 315'
#start = 'adb -s 10.249.243.227:5555 shell input tap 330 500'
slide = 'adb -s 10.249.243.227:5555 shell input swipe 300 1800 850 1000 150'
slide_small = 'adb -s 10.249.243.227:5555 shell input swipe 550 1800 550 1500 100'
slide_back = 'adb -s 10.249.243.227:5555 shell input swipe 550 1100 550 1300 100'

#comment = 'adb -s 192.168.199.219:5555 shell input tap 990 1440'
#close = 'adb -s 192.168.199.219:5555 shell input tap 577 315'
#start = 'adb -s 192.168.199.219:5555 shell input tap 330 500'
#slide = 'adb -s 192.168.199.219:5555 shell input swipe 550 1800 550 1100 100'


t_comment = 'adb -s 10.249.243.227:5555 shell input tap 1000 1407'
t_close = 'adb -s 10.249.243.227:5555 shell input tap 577 315'
t_start = 'adb -s 10.249.243.227:5555 shell input tap 330 500'
t_slide = 'adb -s 10.249.243.227:5555 shell input swipe 550 500 550 -1000 100'


# os.system(adb1)
# time.sleep()

while True:
    #os.system(comment)
    #time.sleep(3.0)
    #os.system(close)
    time.sleep(2.5)
    os.system(slide_back)
    os.system(slide)

