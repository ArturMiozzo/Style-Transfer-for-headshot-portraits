import cv2
import numpy as np
import os
import math

def showStack(stack):
    for image in stack:
        cv2.imshow('stack', image)
        cv2.waitKey(0)

class landmarksDetector:
    def __init__(self):
        LBFmodel = "lbfmodel.yaml"
        haarcascade = "haarcascade_frontalface_alt2.xml"

        self.detector = cv2.CascadeClassifier(haarcascade)

        self.landmark_detector  = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(LBFmodel)
    
    def getLandmarks(self, image):
        face = self.detector.detectMultiScale(image)
        _, landmarks_res = self.landmark_detector.fit(image, face)

        return landmarks_res[0][0]

def warpFace(src, pq1, pq2):
        h, w = src.shape
        trans_coord = np.meshgrid(range(h), range(w), indexing='ij')
        yy, xx = trans_coord[0].astype(np.float64), trans_coord[1].astype(np.float64)  # might need to switch

        xsum = xx * 0.
        ysum = yy * 0.
        wsum = xx * 0.
        for i in range(len(pq1) - 1):
            if i in {16, 21, 26, 30, 35, 47}:
                continue
            elif i == 41:
                j = 36
            elif i == 47:
                j = 42
            elif i == 59:
                j = 48
            elif i == 67:
                j = 60
            else:
                j = i + 1
            # Computes u, v
            p_x1, p_y1 = (pq1[i, 0], pq1[i, 1])
            q_x1, q_y1 = (pq1[j, 0], pq1[j, 1])
            qp_x1 = q_x1 - p_x1
            qp_y1 = q_y1 - p_y1
            qpnorm1 = (qp_x1 ** 2 + qp_y1 ** 2) ** 0.5

            u = ((xx - p_x1) * qp_x1 + (yy - p_y1) * qp_y1) / qpnorm1 ** 2
            v = ((xx - p_x1) * -qp_y1 + (yy - p_y1) * qp_x1) / qpnorm1

            # Computes x', y'
            p_x2, p_y2 = (pq2[i, 0], pq2[i, 1])
            q_x2, q_y2 = (pq2[j, 0], pq2[j, 1])
            qp_x2 = q_x2 - p_x2
            qp_y2 = q_y2 - p_y2
            qpnorm2 = (qp_x2 ** 2 + qp_y2 ** 2) ** 0.5

            x = p_x2 + u * (q_x2 - p_x2) + (v * -qp_y2) / qpnorm2  # X'(x)
            y = p_y2 + u * (q_y2 - p_y2) + (v * qp_x2) / qpnorm2  # X'(y)

            # Computes weights
            d1 = ((xx - q_x1) ** 2 + (yy - q_y1) ** 2) ** 0.5
            d2 = ((xx - p_x1) ** 2 + (yy - p_y1) ** 2) ** 0.5
            d = np.abs(v)
            d[u > 1] = d1[u > 1]
            d[u < 0] = d2[u < 0]
            W = (qpnorm1 ** 1 / (10 + d)) ** 1

            wsum += W
            xsum += W * x
            ysum += W * y

        x_m = xsum / wsum
        y_m = ysum / wsum
        vx = xx - x_m
        vy = yy - y_m
        vx[x_m < 1] = 0
        vx[x_m > w] = 0
        vy[y_m < 1] = 0
        vy[y_m > h] = 0

        vx = (vx + xx).astype(int)
        vy = (vy + yy).astype(int)
        vx[vx >= w] = w - 1
        vy[vy >= h] = h - 1

        warp = np.ones(src.shape)
        warp[yy.astype(int), xx.astype(int)] = src[vy, vx]

        return warp, vx, vy

def denseCorrespondence(input, port, energy_port):
    
    detector = landmarksDetector()

    landmarks_input = detector.getLandmarks(input)
    landmarks_port = detector.getLandmarks(port)

    denseCorrespondence = []

    for image in energy_port:
        denseCorrespondence.append(warpFace(image, landmarks_input, landmarks_port))

    showStack(denseCorrespondence)

    return denseCorrespondence

def computeDecomposition(image, stackSize):

    laplaceStack = []
    energyStack = []

    ksize = stackSize*stackSize

    laplaceStack.append(127 + image - cv2.GaussianBlur(image, (ksize,ksize), 2))

    for i in range(1, stackSize):            
        laplaceStack.append(127+(cv2.GaussianBlur(image, (ksize,ksize), pow(2,i)) - cv2.GaussianBlur(image, (ksize,ksize), pow(2,i+1))))
        
    for i in range(stackSize):     
        energyStack.append(cv2.GaussianBlur(np.square(laplaceStack[i]), (ksize,ksize), pow(2,i+1)))
    
    #showStack(laplaceStack)
    #showStack(energyStack)

    return laplaceStack, energyStack    

def robustTransfer(laplaceStack, energyStack, denseCorrespondence):

    outputStack = []

    for i in range(len(laplaceStack)):
        outputStack.append(laplaceStack[i]*np.sqrt(denseCorrespondence[i]/(energyStack[i]+pow(0.01,2))))
    
    showStack(outputStack)
    
    return outputStack

inputImage = 'jose.jpg'
portImage = 'george.jpg'

image_input = cv2.imread(inputImage)
image_port = cv2.imread(portImage)

input_gray = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
port_gray = cv2.cvtColor(image_port, cv2.COLOR_RGB2GRAY)

input_laplace, input_energy = computeDecomposition(input_gray, 7)
_, port_energy = computeDecomposition(port_gray, 7)

port_correspondence = denseCorrespondence(input_gray, port_gray, port_energy)

robust_transfer = robustTransfer(input_laplace, input_energy, port_correspondence)

image_merge = np.concatenate((image_input, image_port), axis=1)
height, width = image_input.shape[0:2]

#sift = cv2.SIFT_create()
#kp = sift.detect(image_gray,None)
#image=cv2.drawKeypoints(image_gray,kp,image)

cv2.imshow('input',image_merge)
cv2.waitKey(0)