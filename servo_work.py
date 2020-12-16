#!/usr/bin/python
import time
import spidev 
import RPi.GPIO as GPIO
from PCA9685 import PCA9685

pwm = PCA9685()

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 100
spi.mode = 0
angle_x = 90
angle_y = 30
move_angle = 1.5
ALPHA = 0.1
x_filter, y_filter = 512, 512
rotation = 0 

try:
    while True:
    
        position = bytearray(5)
        position = spi.xfer2([0,0,0,0,0])
        x = (position[1] << 8) | position[0]
        y = (position[3] << 8) | position[2]

        x_filter = x*ALPHA + x_filter*(1-ALPHA)
        y_filter = y*ALPHA + y_filter*(1-ALPHA)

        button = (position[4] & 1) | (position[4] & 2)
        
        #print(position)
        print("x",x_filter)
        time.sleep(0.15)
        print("y",y_filter)
        time.sleep(0.15)
        
        pwm.setPWMFreq(50)
                
        # if x_filter > 700:
        #     angle_x -= move_angle
        #     rotation = 1 
        # elif x_filter < 350: 
        #     angle_x += move_angle
        #     rotation = 1
        # else: 
        #     rotation = 0 
        # if y_filter > 700: 
        #     angle_y += move_angle 
        #     rotation = 1
        # elif y_filter < 350: 
        #     angle_y -= move_angle
        #     rotation = 1
        # else: 
        #     rotation = 0

        # if angle_x > 170: 
        #     angle_x = 170
        # elif angle_x < 10: 
        #     angle_x = 10 
    
        # if angle_y > 90: 
        #     angle_y = 90
        # elif angle_y < 10: 
        #     angle_y = 10
        

        # pwm.setRotationAngle(1,angle_x)
        # pwm.setRotationAngle(0,angle_y)

        
        
finally:
    spi.close()
    pwm.exit_PCA9685()
                    
    
