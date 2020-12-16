import RPi.GPIO as GPIO
import time 

buzzPin = 40
GPIO.setmode(GPIO.BOARD)
GPIO.setup(buzzPin,GPIO.OUT)
while(1):
    GPIO.output(buzzPin,True)
    time.sleep(1)
    GPIO.output(buzzPin,False)
    time.sleep(1)

GPIO.output(buzzPin,False)
GPIO.cleanup(7)