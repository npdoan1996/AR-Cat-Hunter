#!/usr/bin/python
import time
import spidev 
import RPi.GPIO as GPIO
from PCA9685 import PCA9685

pwm = PCA9685()

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 100000
spi.mode = 0
angle_x = 90
angle_y = 90 

try:
    while True:
    
        position = bytearray(5)
        position = spi.xfer2([0x00,0x00,1000000, 10, 8])
        x = (position[1] << 8) | position[0]
        y = (position[3] << 8) | position[2]

        button = (position[4] & 1) | (position[4] & 2)
        
        print(position)
        print("x",x)
        time.sleep(0.000015)
        print("y",y)
        time.sleep(0.000015)
        
        pwm.setPWMFreq(50)
                
        if x > 700:
            angle_x -= 1 
        elif x < 350: 
            angle_x += 1
        if y > 700: 
            angle_y += 1 
        elif y < 350: 
            angle_y -= 1 
        
        pwm.setRotationAngle(1,angle_x)
        pwm.setRotationAngle(0,angle_y)

        if angle_x > 170: 
            angle_x = 170
        elif angle_x < 10: 
            angle_x = 10 
    
        if angle_y > 90: 
            angle_y = 90
        elif angle_y < 10: 
            angle_y = 10

        
        
finally:
    spi.close()
    pwm.exit_PCA9685()
                    
    
