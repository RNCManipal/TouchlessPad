import cv2
import numpy as np
import time
import mouse
import mediapipe as mp
import math

#HAND TRACKING MODULE ATTACHED
class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
    # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
    # totalFingers = fingers.count(1)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

######################
wCam, hCam = 640, 480
frameR = 100     #Frame Reduction
smoothening = 7  #random value
######################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0)
stop = 0
cap.set(3, wCam)
cap.set(4, hCam)
wScr, hScr = 1280, 700
text = ''
close = 0

detector = handDetector()
#

# print(wScr, hScr)

frame_rate = 10
prev = 0

while True:
    # Step1: Find the landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # Step2: Get the tip of the index and middle finger
    if len(lmList) != 0:
        x0, y0 = lmList[4][1:]
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x3, y3 = lmList[16][1:]
        x4, y4 = lmList[20][1:]
        # Step3: Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)
        # Step4: Only Index Finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # Step5: Convert the coordinates
            xi = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            yi = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # Step6: Smooth Values
            clocX = plocX + (xi - plocX) / smoothening
            clocY = plocY + (yi - plocY) / smoothening
            # Step7: Move Mouse
            mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        if fingers[1] == 0 and fingers [2] == 0 and fingers[4]:
            stop+=1
            if stop>20:
                mouse.right_click()
                stop = 0
        # Step8: Both Index and middle are up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Step9: Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            stop+=1
            # Step10: Click mouse if distance short
            if length < 40 and stop>20:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                mouse.click(button='left')
                stop = 0
           #elif length>40:
           #    ##FOR DRAGGING WHILE USING A LOCAL CANVAS APP
           #    xi = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
           #    yi = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
           #    # Step6: Smooth Values
           #    clocX = plocX + (xi - plocX) / smoothening
           #    clocY = plocY + (yi - plocY) / smoothening
           #    plocX, plocY = clocX, clocY
           #    pyautogui.dragTo(wScr - clocX, clocY)
           #    stop = 0
        if fingers[1] and fingers[2] and fingers[3]:
            stop+=1
            if stop>20:
                mouse.double_click(button='left')
                stop = 0
        if all(x == 0 for x in fingers):
            if (y1>=(hCam/2)):
                mouse.wheel(delta=-1)
            else:
                mouse.wheel(delta=1)
        # ALT + F4 to clopse app    
        #if all(x == 1 for x in fingers):
        #    text = "Keep fingers up to confirm"
        #    close += 1
        #    if close >= 50:
        #        pyautogui.hotkey('alt', 'f4')
        #        close=0
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # Step12: Display
    img1 = cv2.flip(img, 1)
    cv2.putText(img1, str(fps), (28, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)
    cv2.imshow("Image", img1)
    if (cv2.waitKey(1) & 0xFF == ord('d')):
            break
cap.release()
cv2.destroyAllWindows()
