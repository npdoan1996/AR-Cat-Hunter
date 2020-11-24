import cv2
import numpy as np
import math
import random
print(cv2.__version__)

# global variables
dispW = 640
dispH = 480
flip = 2
num_of_cat = 4
crosshair_x = int(dispW/2-64)
crosshair_y = int(dispH/2-64)
counter = 0 
evt = -1

def click(event,x,y,flag,params): 
    global pnt
    global evt
    if event==cv2.EVENT_LBUTTONDOWN:
        print('Mouse Clicked')
        print(x,',',y)
        pnt=(x,y)
        evt=event

cv2.namedWindow('colorDetection')
cv2.setMouseCallback('colorDetection',click)
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

def drawAliveCat(location):
    x,y = location 
    if dispW - x > 100 and dispH - y > 100: 
        roi = frame[y:y+100, x:x+100] #The area of roi, cropping the frame 
        img_bg = cv2.bitwise_and(roi, roi, mask=cat_mask_inv)
        img_fg = cv2.bitwise_and(cat, cat, mask=cat_mask)
        dst = cv2.add(img_bg, img_fg)
        frame[y:y + 100, x:x + 100] = dst

def drawDeadCat(location):
    x,y = location 
    if dispW - x > 200 and dispH - y > 100: 
        roi = frame[y:y+100, x:x+200] #The area of roi, cropping the frame 
        img_bg = cv2.bitwise_and(roi, roi, mask=dead_cat_mask_inv)
        img_fg = cv2.bitwise_and(dead_cat, dead_cat, mask=dead_cat_mask)
        dst = cv2.add(img_bg, img_fg)
        frame[y:y + 100, x:x + 200] = dst

def drawCrosshair(x,y): 
    ROI = frame[y:y+128, x:x+128] #The area of roi, cropping the frame 
    ROIBG = cv2.bitwise_and(ROI,ROI,mask=crosshair_BGMask)
    frame[y:y + 128, x:x + 128] = ROIBG

def isCollision(cat_x,cat_y):
    distance = math.sqrt(math.pow(cat_x - crosshair_x, 2) + math.pow(cat_y - crosshair_y, 2))
    if distance < 27: 
        return True
    else:
        return False  

class Cat: 
    def __init__(self, location, alive, display):
        self.location = location
        self.alive = alive
        self.display = display

#loading alive cat image and making its mask to overlay on the video feed
cat = cv2.imread("images/cat.png",-1)
cat_mask = cat[:,:,3]
cat_mask_inv = cv2.bitwise_not(cat_mask)
cat = cat[:,:,0:3]

# resizing alive cat image
cat = cv2.resize(cat,(100,100),interpolation=cv2.INTER_AREA)
cat_mask = cv2.resize(cat_mask,(100,100),interpolation=cv2.INTER_AREA)
cat_mask_inv = cv2.resize(cat_mask_inv,(100,100),interpolation=cv2.INTER_AREA)

#loading dead cat image and making its mask to overlay on the video feed
dead_cat = cv2.imread("images/dead_cat.png",-1)
dead_cat_mask = dead_cat[:,:,3]
dead_cat_mask_inv = cv2.bitwise_not(dead_cat_mask)
dead_cat = dead_cat[:,:,0:3]

# resizing dead cat image
dead_cat = cv2.resize(dead_cat,(200,100),interpolation=cv2.INTER_AREA)
dead_cat_mask = cv2.resize(dead_cat_mask,(200,100),interpolation=cv2.INTER_AREA)
dead_cat_mask_inv = cv2.resize(dead_cat_mask_inv,(200,100),interpolation=cv2.INTER_AREA)

#loading crosshair image and making its mask to overlay on the video feed
crosshair = cv2.imread("images/crosshair.png",-1)
crosshair = cv2.resize(crosshair,(128,128))
crosshairGray = cv2.cvtColor(crosshair, cv2.COLOR_BGR2GRAY)
_,crosshair_BGMask = cv2.threshold(crosshairGray,225,255,cv2.THRESH_BINARY)
crosshair_FGMask = cv2.bitwise_not(crosshair_BGMask) 

alive_cats = []
dead_cats = []
for i in range(num_of_cat):
    x = random.randint(0, dispW - 100)
    y = random.randint(0, dispH-100)
    alive_cats.append(Cat((x,y), True, False))

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
    
    for i in range(num_of_cat):
        alive_cats[i] = Cat((0,0), True, False)

    i = 0 
    for area in detected_areas: 
        alive_cats[i] = Cat(area, True, True)
        i+=1

    for alive_cat in alive_cats: 
        collision = False
        if evt == 1: 
            collision = isCollision(alive_cat.location[0], alive_cat.location[1])
            print(collision)
            evt = -1
        if collision: 
            alive_cat.display = False
            dead_cats.append(Cat(alive_cat.location, False, True))
        distance = 100000
        for d_cat in dead_cats: 
            new_distance = math.sqrt(math.pow(d_cat.location[0] - alive_cat.location[0], 2) + math.pow(d_cat.location[1] - alive_cat.location[1], 2))
            if distance > new_distance: 
                distance = new_distance
            if distance < 100: 
                alive_cat.display = False
                break

        if alive_cat.display != False: 
            drawAliveCat(alive_cat.location)

    counter+=1
    if counter == 150: 
        counter = 0 
        if not dead_cats:
            continue 
        else: 
           dead_cats.pop(0); 

    for d_cat in dead_cats:
        drawDeadCat(d_cat.location)
    
    drawCrosshair(crosshair_x,crosshair_y)

    cv2.imshow('orange_mask', orange_mask)
    cv2.moveWindow('orange_mask',0,500)
    cv2.imshow('colorDetection',frame)
    cv2.moveWindow('colorDetection',0,0)



    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
