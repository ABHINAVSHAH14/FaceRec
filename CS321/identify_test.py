
#-------------------------------------------------------------------------------------------
# MODULES
#-------------------------------------------------------------------------------------------

import os
import cv2
import sys
import time
import psutil
import shutil
import timeit
import smtplib
import numpy as np
from PIL import Image
import tensorflow as tf

# Clean up previous files

if (os.path.exists(os.getcwd() + "/target/analyze.jpg")):
    os.remove(os.getcwd() + "/target/analyze.jpg")
if (os.path.exists(os.getcwd() + "/target/imposter.jpg")):
    os.remove(os.getcwd() + "/target/imposter.jpg")


#-------------------------------------------------------------------------------------------
# OPENCV VIDEO CAPTURE
#-------------------------------------------------------------------------------------------


def face():

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(os.getcwd() + '/data/Face.xml')

    collected = 0


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
                cv2.imwrite("target/analyze.jpg", img[y:y+h,x:x+w])
                collected += 1
                break

        if (collected >= 10):
            cam.release()
            cv2.destroyAllWindows()
            print("\nAnalyzing...\n")
            break

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    Image.open(os.getcwd() + "/target/analyze.jpg").show()
    
             
#-------------------------------------------------------------------------------------------
# TENSORFLOW INITIALIZATION CONNECTED TO OPENCV
#-------------------------------------------------------------------------------------------


    path = "target/analyze.jpg"

    image = tf.gfile.FastGFile(path, 'rb').read()

    label= [line.rstrip() for line in tf.gfile.GFile("/data/retrained_labels.txt")]

    with tf.gfile.FastGFile("/data/retrained_graph.pb", 'rb') as f:
        graph = tf.GraphDef()
        graph.ParseFromString(f.read())
        tf.import_graph_def(graph, name='')

    with tf.Session() as sess:
        tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(tensor, \
                 {'DecodeJpeg/contents:0': image})
        
        sort = predictions[0].argsort()[-len(predictions[0]):][::-1]
        #print(str(sort[0]))


#-------------------------------------------------------------------------------------------
# RUNS DATA 
#-------------------------------------------------------------------------------------------
        
    users = []

    with open('./data/retrained_labels.txt') as file:
        for line in file:
            users.append(line)

    # i = 0
    # while i < len(users):
    #     print(users[i] + str(predictions[0][i] * 100)[0:5] + "%")
    #     i += 1

    # percent = str(predictions)
    # print(percent)

    # percent = str(predictions[0][0] * 100)
    # print(percent[0:5] + "%\n")

    cam.release()
    cv2.destroyAllWindows()

    

face()

#-------------------------------------------------------------------------------------------
# END OF CODE
#-------------------------------------------------------------------------------------------
