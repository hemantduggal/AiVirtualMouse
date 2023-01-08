import cv2
import HandTrackingModule as htm
import time
import autopy
import pyautogui
import numpy as np

wCam, hCam = 640, 480
frameR = 100 #Frame Reduction
smoothening = 7
wScreen, hScreen = autopy.screen.size()
toggle = False
startTime = 0
count = 0
# wScreen, hScreen = pyautogui.size()
# print(pwScreen,phScreen , wScreen,hScreen)
# print(wScreen,hScreen)
currTime = 0
prevTime = 0
prevLocX, prevLocY = 0,0
currLocX, currLocY = 0,0

cam = cv2.VideoCapture(0)

cam.set(3, wCam) #Width
cam.set(4, hCam) #Height

detector = htm.HandDetector(maxHands=1, minDetectionConfidence=0.8)

def moveMouse(x1,y1):
    # 5.Convert Coordinates
    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScreen))
    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))

    # 6.Smoothing Values
    currLocX = prevLocX + (x3 - prevLocX) / smoothening
    currLocY = prevLocY + (y3 - prevLocY) / smoothening

    # 7.Move Mouse
    autopy.mouse.move(currLocX, currLocY)
    # pyautogui.moveTo(currLocX, currLocY)
    return currLocX, currLocY

while True:
    #1.Find Hand Landmarks
    success, img = cam.read()

    img = cv2.flip(img,1)
    img = detector.findHands(img)
    landmarksList, boundingBox = detector.findPosition(img, draw=True)

    #2.Get the tip of the index and middle finger
    if len(landmarksList) != 0:
        x1, y1 = landmarksList[8][1:]
        x2, y2 = landmarksList[12][1:]
        # print(x1,y1,x2,y2)

        #3.Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (68, 150, 253), 2)

        #4.Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            prevLocX, prevLocY = moveMouse(x1,y1)
            cv2.circle(img, (x1,y1), 15, (6, 47, 235),cv2.FILLED)

        #8.Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9.Find distance between fingers
            length, img, lineInfo = detector.findDistance(8,12,img)
            # print(length)
            # 10.Click mouse if distance short
            if length < 40:
                if count < 1:
                    startTime = time.time()
                    count += 1

                if count == 1:
                    if int(time.time() - startTime) >= 2 and toggle == False:
                        toggle = True
                        autopy.mouse.toggle(down=toggle)
                        # print('drag', time.time() - startTime)

                    if toggle:
                        prevLocX, prevLocY = moveMouse(x1,y1)

                cv2.circle(img, (lineInfo[4],lineInfo[5]), 15,
                                              (6, 47, 235),cv2.FILLED)
                # pyautogui.click()
            else:
                if count == 1:
                    toggle = False
                    autopy.mouse.toggle(down=toggle)

                    if (time.time() - startTime) < 2:
                        autopy.mouse.click()
                        # print('click')

                    startTime = 0
                    count = 0
        # print(fingers)
        if fingers.count(1) == 0:
            pyautogui.click(button = "right")




    #11.Frame Rate

    currTime = time.time()
    fps = 1/(currTime - prevTime)
    fps = str(int(fps))
    prevTime = currTime

    cv2.putText(img, "FPS: {}".format(fps), (10,35), cv2.FONT_HERSHEY_PLAIN,
                                                    2, (0, 0, 255), 2)

    #12.Display
    cv2.imshow("AI Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


