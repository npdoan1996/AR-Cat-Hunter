import cv2
import numpy as np
print(cv2.__version__)

dispW = 640
dispH = 480
flip = 2

# Uncomment These next Two Line for Pi Camera
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)

# cam=cv2.VideoCapture(0)  #use this line for webcame(try 0 or 1)

kernel_open = np.ones((4,4),np.uint8)
kernel_dilate = np.ones((10,10),np.uint8)
kernel_close = np.ones((10,10),np.uint8)
low_orange = np.array([0,100,100])
high_orange = np.array([20,255,255])
# low_orange2 = np.array([0,100,100])
# high_orange2 = np.array([30,255,255])

def drawContourRect(): 
    contours,hierarchy = cv2.findContours(orange_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=lambda x:cv2.contourArea(x), reverse=True)
    contours = contours[:4]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x,y,w,h) = cv2.boundingRect(cnt)
        if area >= 200: 
            #cv2.drawContours(frame,[cnt],0,(255,0,0),3)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


while True:
    ret, frame = cam.read()

    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Orange detection
    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
    orange_mask = cv2.morphologyEx(orange_mask,cv2.MORPH_OPEN,kernel_open)
    # orange_mask = cv2.dilate(orange_mask, kernel_dilate, iterations=1)
    orange_mask = cv2.morphologyEx(orange_mask,cv2.MORPH_CLOSE,kernel_close)
    orange = cv2.bitwise_and(frame, frame, mask=orange_mask)
     
    drawContourRect()

    cv2.imshow('orange_mask', orange_mask)
    cv2.moveWindow('orange_mask',0,500)
    cv2.imshow('colorDetection',frame)
    cv2.moveWindow('colorDetection',0,0)
    


    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
