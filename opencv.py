# Importing packages
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import shutil
import os


def nothing(x):
    pass

# General Variable values
cv2.namedWindow("Configurations")
cv2.createTrackbar("Pen Size", "Configurations", 5, 100, nothing)
cv2.createTrackbar("Eraser Size", "Configurations", 30, 200, nothing)
imgCanvas = np.ones((576, 1000, 3), np.uint8)
imgCanvas[:] = 255
xp = 0
yp = 0
drawColor= (255, 0, 255)
eraserColor=(255,255,255)
n=0
path = r'C:\Users\jason\Desktop\Projects\Vedantu_Non-touch'
pathNew = r'C:\Users\jason\Desktop\Projects\Vedantu_Non-touch\Output'
if not os.path.exists("Output"):
    os.mkdir("Output")
else:
    shutil.rmtree("Output")
    os.mkdir("Output")


# Modules
class handTracker():
    def __init__(self, mode=False, maxHands=2,complexity = 1, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=False):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 15, drawColor, cv2.FILLED)

        return lmlist


# Main Function
def main():
    global imgCanvas
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    xp = 0
    yp = 0
    alpha =0.7
    stop=0
    n=0
    path = r'C:\Users\jason\Desktop\Projects\Vedantu_Non-touch'
    pathNew = r'C:\Users\jason\Desktop\Projects\Vedantu_Non-touch\Output'

    tracker = handTracker()
    tipIds = [4, 8, 12, 16, 20]

    while True:
        success, image = cap.read()
        image = cv2.flip(image, 1)

        brushThickness = cv2.getTrackbarPos("Pen Size", "Configurations")
        eraserThickness = cv2.getTrackbarPos("Eraser Size", "Configurations")

        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
           # fingers = tracker.fingersUp()


            fingers = []
            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            if fingers[1] and fingers[2]:
                if fingers[3]:
                    cv2.circle(image, (x2, y2), 15, drawColor, cv2.FILLED)
                    print("Erasing Mode")
                    if xp == 0 and yp == 0:
                        xp, yp = x2, y2
                    cv2.line(image, (x2, y2), (x2, y2), eraserColor, eraserThickness)
                    cv2.line(imgCanvas, (x2, y2), (x2, y2), eraserColor, eraserThickness)
                    xp, yp = x1, y1
                else:
                    xp, yp = 0, 0
                    print("Selection Mode")
                    cv2.rectangle(image, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            if fingers[1] and fingers[2] == False:
                cv2.circle(image, (x1, y1), 15, drawColor, cv2.FILLED)
                print("Drawing Mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(image, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp,yp=x1,y1


            if all(x >= 1 for x in fingers):
                stop=stop+1
                if stop==38:
                    saved = Image.fromarray(imgCanvas)
                    name = str(n) + '.jpg'
                    slash = str('/')
                    source = path +slash+ name
                    saved.save(name)
                    shutil.move(source, pathNew)
                    n = n + 1
                    imgCanvas = np.ones((576, 1000, 3), np.uint8)
                    imgCanvas[:] = 255
                    stop=0


        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        imgEx1 = image[0 : 576, 0: 1000]
        imgEx = cv2.bitwise_and(imgEx1, imgInv)
        imgEx = cv2.bitwise_or(imgEx, imgCanvas)
        imgEx = cv2.addWeighted(imgEx1, alpha, imgCanvas, 1 - alpha, 1.0)
        image[0 : 576, 0: 1000] = imgEx


        cv2.imshow("Video", image)
        cv2.imshow("Canvas", imgCanvas)
        if (cv2.waitKey(1) & 0xFF == ord('d')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
