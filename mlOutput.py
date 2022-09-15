import os
import cv2

path = os.getcwd()


path = path + '\Output'
n = 0

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
        #To show which image is being shown.
        #cv2.imshow('img', img)
        
        #Insert ML code here ('img' is the newest image in the Output folder. 'pathNew' is the images path) - 


        if (cv2.waitKey(1) & 0xFF == ord('d')):
            break
