import cv2
import numpy as np
import math
print(cv2.__version__)

# global variables
dispW = 640
dispH = 480
flip = 2
numOfCat = 0

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

def colorDetect(): 
    contours,hierarchy = cv2.findContours(orange_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=lambda x:cv2.contourArea(x), reverse=True)
    contours = contours[:4]
    location = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x,y,w,h) = cv2.boundingRect(cnt)
        if area >= 300: 
            #cv2.drawContours(frame,[cnt],0,(255,0,0),3)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            location.append((int(x),int(y)))
    return location

def drawCat(location):
    x,y = location 
    if dispW - x > 100 and dispH - y > 100: 
        roi = frame[y:y+100, x:x+100] #The area of roi, cropping the frame 
        img_bg = cv2.bitwise_and(roi, roi, mask=cat_mask_inv)
        img_fg = cv2.bitwise_and(cat, cat, mask=cat_mask)
        dst = cv2.add(img_bg, img_fg)
        frame[y:y + 100, x:x + 100] = dst

#loading cat image and making its mask to overlay on the video feed
cat = cv2.imread("images/cat.png",-1)
cat_mask = cat[:,:,3]
cat_mask_inv = cv2.bitwise_not(cat_mask)
cat = cat[:,:,0:3]

# resizing cat image
cat = cv2.resize(cat,(100,100),interpolation=cv2.INTER_AREA)
cat_mask = cv2.resize(cat_mask,(100,100),interpolation=cv2.INTER_AREA)
cat_mask_inv = cv2.resize(cat_mask_inv,(100,100),interpolation=cv2.INTER_AREA)
cat_positions = []

while True:
    ret, frame = cam.read()

    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Orange detection
    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
    orange_mask = cv2.morphologyEx(orange_mask,cv2.MORPH_OPEN,kernel_open)
    # orange_mask = cv2.dilate(orange_mask, kernel_dilate, iterations=1)
    orange_mask = cv2.morphologyEx(orange_mask,cv2.MORPH_CLOSE,kernel_close)
    orange = cv2.bitwise_and(frame, frame, mask=orange_mask)
     
    detected_areas = colorDetect()    

    for position in detected_areas: 
        drawCat(position)


    cv2.imshow('orange_mask', orange_mask)
    cv2.moveWindow('orange_mask',0,500)
    cv2.imshow('colorDetection',frame)
    cv2.moveWindow('colorDetection',0,0)
    


    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
