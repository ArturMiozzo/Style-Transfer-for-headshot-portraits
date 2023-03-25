import cv2
import numpy as np
import os
import math
import scipy
import skimage

def showStack(stack):
    for image in stack:
        cv2.imshow('stack', ((0.5+image) - (0.5+image).min()) / ((0.5+image).max() - (0.5+image).min()))
        cv2.waitKey(0)

def normalize(img):
    return img / 255

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

def warpFace(image, source_points, target_points):
    imh, imw = image.shape
    out_image = image.copy()
    xs = np.arange(imw)
    ys = np.arange(imh)
    interpFN = scipy.interpolate.interp2d(xs, ys, image)

    tri = scipy.spatial.Delaunay(source_points)

    for triangle_indices in tri.simplices:

        source_triangle = source_points[triangle_indices]
        target_triangle = target_points[triangle_indices]

        source_matrix = np.row_stack((source_triangle.transpose(), (1, 1, 1)))
        target_matrix = np.row_stack((target_triangle.transpose(), (1, 1, 1)))
        A = np.matmul(target_matrix, np.linalg.inv(source_matrix))

        A_inverse = np.linalg.inv(A)

        tri_rows = target_triangle.transpose()[1]
        tri_cols = target_triangle.transpose()[0]

        row_coordinates, col_coordinates = skimage.draw.polygon(tri_rows, tri_cols)

        for x, y in zip(col_coordinates, row_coordinates):
            #point inside target triangle mesh
            point_in_target = np.array((x, y, 1))

            #point inside source image
            point_on_source = np.dot(A_inverse, point_in_target)

            x_source = point_on_source[0]
            y_source = point_on_source[1]

            source_value = interpFN(x_source, y_source)
            try:
                out_image[y, x] = source_value
            except IndexError:
                continue

    #cv2.imshow('in', ((0.5+image) - (0.5+image).min()) / ((0.5+image).max() - (0.5+image).min()))
    #cv2.imshow('out', ((0.5+out_image) - (0.5+out_image).min()) / ((0.5+out_image).max() - (0.5+out_image).min()))
    #cv2.waitKey(0)

    return out_image

def denseCorrespondence(energy_port, landmarks_input, landmarks_port):
    
    denseCorrespondence = []

    for image in energy_port:
        denseCorrespondence.append(warpFace(image, landmarks_port, landmarks_input))

    #showStack(denseCorrespondence)

    return denseCorrespondence

def computeDecomposition(image, stackSize):

    laplaceStack = []
    energyStack = []
    residualStack = []

    imageNorm = normalize(image)
  
    ksize = 5*2 + 1

    laplaceStack.append(imageNorm - cv2.GaussianBlur(imageNorm, (ksize,ksize), 2))

    for i in range(1, stackSize):       
        ksize = 5*pow(2,i) + 1
        laplaceStack.append(cv2.GaussianBlur(imageNorm, (ksize,ksize), pow(2,i)) - cv2.GaussianBlur(imageNorm, (5*pow(2,i+1) + 1,5*pow(2,i+1) + 1), pow(2,i+1)))
        
    for i in range(stackSize):    
        ksize = 5*pow(2,i+1) + 1
        energyStack.append(cv2.GaussianBlur(np.square(laplaceStack[i]), (ksize,ksize), pow(2,i+1)))
    
    ksize = 5*pow(2,stackSize) + 1
    residual = cv2.GaussianBlur(imageNorm, (ksize,ksize), pow(2,stackSize))
    
    #showStack(laplaceStack)
    #showStack(energyStack)

    return laplaceStack, energyStack, residual

def robustTransfer(laplaceStack, energyStack, denseCorrespondence):

    outputStack = []
    ksize = len(laplaceStack)*len(laplaceStack)

    for i in range(len(laplaceStack)):
        gain = np.sqrt((denseCorrespondence[i])/((energyStack[i])+pow(0.01,2)))
        gain[gain > 2.8] = 2.8
        gain[gain < 0.9] = 0.9  

        ksize = 5*pow(2,i+1) + 1
        outputStack.append((laplaceStack[i])*cv2.GaussianBlur(gain, (ksize,ksize), 3*pow(2,i)))
    
    #showStack(outputStack)
    
    return outputStack

def sumStack(image, stack):

    final_image = normalize(image)

    for img in stack:
        final_image += img
    
    final_image = (final_image - final_image.min()) / (final_image.max() - final_image.min())

    return (final_image * 255).astype(np.uint8)

inputImage = 'farido.png'
portImage = 'turing.jpeg'

image_input = cv2.imread(inputImage)
image_port = cv2.imread(portImage)

detector = landmarksDetector()

landmarks_input = detector.getLandmarks(image_input)
landmarks_port = detector.getLandmarks(image_port)

if len(image_input.shape)>2:
    channel_count = image_input.shape[2]
else:
    channel_count = 1

output = []

for channel in range(channel_count):

    input_gray = cv2.split(image_input)[channel]
    port_gray = cv2.split(image_port)[channel]

    input_laplace, input_energy, input_residual = computeDecomposition(input_gray, 7)
    _, port_energy, port_residual = computeDecomposition(port_gray, 7)

    port_correspondence = denseCorrespondence(port_energy, landmarks_input, landmarks_port)

    robust_transfer = robustTransfer(input_laplace, input_energy, port_correspondence)
    
    robust_transfer.append(warpFace(port_residual, landmarks_port, landmarks_input))

    output.append(sumStack(input_gray, robust_transfer))

completeOutput = cv2.merge(output)

#image_merge = np.concatenate((input_gray, output), axis=1)
#height, width = image_input.shape[0:2]

#sift = cv2.SIFT_create()
#kp = sift.detect(image_gray,None)
#image=cv2.drawKeypoints(image_gray,kp,image)

cv2.imshow('input', image_input)
cv2.imshow('port', image_port)
cv2.imshow('style', completeOutput)
cv2.waitKey(0)