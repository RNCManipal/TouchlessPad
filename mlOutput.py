import os
import cv2

n = 0
path = r"C:\Users\jason\Desktop\Projects\Vedantu_Non-touch\Output"

while True:
    if not os.path.exists("Output/0.jpg"):
        print("No Output folder")
    else:
        name = str(n) + '.jpg'
        nameNext = str(n+1) + '.jpg'
        pathNew = os.path.join(path, name)
        pathNext = os.path.join(path, nameNext)
        if os.path.exists(pathNext):
            n = n+1
        img = cv2.imread(pathNew)
        print (pathNew)
        cv2.imshow('img', img)

        if (cv2.waitKey(1) & 0xFF == ord('d')):
            break


