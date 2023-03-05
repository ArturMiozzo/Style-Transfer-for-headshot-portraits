import cv2
import os

inputImage = 'farido.png'
# save facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"
# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"

# check if file is in working directory
if not LBFmodel in os.listdir(os.curdir):
    print("landmarks model not found")
    exit()

# chech if file is in working directory
if not haarcascade in os.listdir(os.curdir):
    print("face model not found")
    exit()

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade)

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

image = cv2.imread(inputImage)

# convert image to Grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Detect faces using the haarcascade classifier on the "grayscale image"
faces = detector.detectMultiScale(image_gray)

# Detect landmarks on "image_gray"
_, landmarks = landmark_detector.fit(image_gray, faces)

for landmark in landmarks:
    for x,y in landmark[0]:
		# display landmarks on "image_rgb"
		# with white colour in BGR and thickness 1
        cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 1)

cv2.imshow('input',image)
cv2.waitKey(0)