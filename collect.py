import os
import cv2
import numpy as np
from PIL import Image


user = raw_input('Enter Name: ')

if (os.path.exists(os.getcwd() + "/images/" + str(user)) == False):
    os.mkdir("images/" + user)
    numImages = 0

else:
    directory = os.listdir(os.getcwd() + "/images/" + str(user))
    numImages = len(directory)
    print(numImages)

directory2 = os.listdir(os.getcwd() + "/rejects")
rejects = len(directory2)

def collect(numImages, rejects):

    collected = 0

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(os.getcwd() + '/data/Face.xml')

    while(True):

        ret, img = cam.read()
        faces = detector.detectMultiScale(img, 1.3, 5)

        for (x,y,w,h) in faces:
            img = cv2.rectangle(img, (x, y),(x + w, y + h), (255, 0, 0), 2)

            cv2.imwrite("target/display/capture.jpg", img[y:y+h,x:x+w])
            display = cv2.imread("target/display/capture.jpg")
            show = cv2.resize(display, (500, 500))
            cv2.imshow('Target', show)
            cv2.imshow('frame',img)   
            
            im = Image.open(os.getcwd() + "/target/display/capture.jpg")
            height, width = im.size

            if (height >= 100):
                cv2.imwrite(os.getcwd() + "/images/" + user + "/" + str(numImages) + ".jpg", img[y:y+h,x:x+w])
                numImages += 1
                collected += 1
            else:
                cv2.imwrite(os.getcwd() + "/rejects/" + str(rejects) + ".jpg", img[y:y+h,x:x+w])
                rejects += 1


        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        if (collected > 100):
            break


        print(numImages)

    cam.release()
    cv2.destroyAllWindows()


collect(numImages, rejects)
    

