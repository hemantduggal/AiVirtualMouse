import cv2
import mediapipe as mp
import time
import math


class HandDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1,
                 minDetectionConfidence=0.5, minTrackingConfidence=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.minDetectionConfidence, self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNumber=0, draw=True):
        xList = []
        yList = []
        self.boundingBox = []
        self.landMarksList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)

                height, width, channel = img.shape
                # print(height,width,channel)

                cx, cy = int(lm.x * width), int(lm.y * height)
                # print(id ,cx , cy)
                xList.append(cx)
                yList.append(cy)

                self.landMarksList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 7, (36, 202, 249), cv2.FILLED)

            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)

            self.boundingBox = xMin, yMin, xMax, yMax

            if draw:
                cv2.rectangle(img, (self.boundingBox[0] - 20, self.boundingBox[1] - 20),(self.boundingBox[2] + 20,
                                                                                         self.boundingBox[3] + 20), (129, 222, 38), 2)
        return self.landMarksList, self.boundingBox

    def fingersUp(self):
        fingers = []

        #Thumb
        if self.landMarksList[self.tipIds[0]][1] < self.landMarksList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #4 Fingers
        for id in range(1,5):
            if self.landMarksList[self.tipIds[id]][2] < self.landMarksList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.landMarksList[p1][1], self.landMarksList[p1][2]
        x2, y2 = self.landMarksList[p2][1], self.landMarksList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) //2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (6, 47, 235), 2)
            cv2.circle(img, (x1,y1), 15, (64, 54, 47), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (64, 54, 47), cv2.FILLED)
            cv2.circle(img, (cx, cy), 15, (64, 54, 47), cv2.FILLED)

        length = math.hypot(x2 - x1 , y2 - y1)
        return length,img,[x1,y1,x2,y2,cx,cy]

def main():
    prevTime = 0
    currTime = 0

    cam = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        success, img = cam.read()

        img = detector.findHands(img)
        landmarksList = detector.findPosition(img)

        if (len(landmarksList) != 0):
            print(landmarksList[4])

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()