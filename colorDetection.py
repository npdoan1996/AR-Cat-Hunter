import cv2
import numpy as np
print(cv2.__version__)

dispW = 640
dispH = 480
flip = 2
# Uncomment These next Two Line for Pi Camera
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)

img1 = np.zeros((100,50,1), np.uint8)
cv2.imshow('img1',img1)
cv2.moveWindow('img1',700,0)
# Or, if you have a WEB cam, uncomment the next line
# (If it does not work, try setting to '1' instead of '0')
# cam=cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Red color 
    low_red = np.array([161,155,84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame,frame, mask=red_mask)
    
    # Blue color 
    low_blue = np.array([94,80,2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    cv2.imshow('nanoCam',frame)
    cv2.moveWindow('nanoCam',0,0)
    cv2.imshow('redMask',blue)
    cv2.moveWindow('redMask',0,500)
    


    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()