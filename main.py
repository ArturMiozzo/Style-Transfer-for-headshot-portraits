import cv2
import numpy as np
import os

def computeLaplaceStack(image, stackSize, gaussSize):
    
    laplaceStack = []
    laplaceStack.append(image - cv2.GaussianBlur(image,(gaussSize,gaussSize),2))
    for i in range(1, stackSize):
        laplaceStack.append(cv2.GaussianBlur(image,(gaussSize,gaussSize),pow(2,i)) - cv2.GaussianBlur(image,(gaussSize,gaussSize),pow(2,i+1)))

    for image in laplaceStack:
        cv2.imshow("laplace", image)
        cv2.waitKey(0)


inputImage = 'farido.png'
portImage = 'gigachad.png'

LBFmodel = "lbfmodel.yaml"
haarcascade = "haarcascade_frontalface_alt2.xml"

if not LBFmodel in os.listdir(os.curdir):
    print("landmarks model not found")
    exit()

if not haarcascade in os.listdir(os.curdir):
    print("face model not found")
    exit()

detector = cv2.CascadeClassifier(haarcascade)

landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

image_input = cv2.imread(inputImage)
image_port = cv2.imread(portImage)

input_gray = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
port_gray = cv2.cvtColor(image_port, cv2.COLOR_RGB2GRAY)

computeLaplaceStack(input_gray, 4, 3)

face_input = detector.detectMultiScale(input_gray)
face_port = detector.detectMultiScale(port_gray)

_, landmarks_input = landmark_detector.fit(input_gray, face_input)
_, landmarks_port = landmark_detector.fit(port_gray, face_port)

image_merge = np.concatenate((image_input, image_port), axis=1)
height, width = image_input.shape[0:2]

for ind in range(min(len(landmarks_input),len(landmarks_port))):
    for i in range(min(len(landmarks_input[ind][0]),len(landmarks_port[ind][0]))):
        x1 = landmarks_input[ind][0][i][0]
        y1 = landmarks_input[ind][0][i][1]
        x2 = landmarks_port[ind][0][i][0]
        y2 = landmarks_port[ind][0][i][1]
        
        cv2.line(image_merge, (int(x1), int(y1)), (int(x2)+width, int(y2)), (0, 255, 0), thickness=1)

#sift = cv2.SIFT_create()
#kp = sift.detect(image_gray,None)
#image=cv2.drawKeypoints(image_gray,kp,image)

cv2.imshow('input',image_merge)
cv2.waitKey(0)