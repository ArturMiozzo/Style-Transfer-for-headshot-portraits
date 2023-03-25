import cv2
import numpy as np
import os
import math
import scipy
import skimage
import glob

def writeStack(stack, path):
    
    global batchName
    
    directory = path+'_'+batchName
    if not os.path.exists(directory):
        os.makedirs(directory)

    index = len(glob.glob(directory+'\*.png'))
    for image in stack:
        cv2.imwrite(directory+'\\'+str(index)+'.png', 255*(((0.5+image) - (0.5+image).min()) / ((0.5+image).max() - (0.5+image).min())))
        index+=1

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
        # para cada imagem na pilha, faz a transformação da face do retrato com a face da entrada
        denseCorrespondence.append(warpFace(image, landmarks_port, landmarks_input))

    writeStack(denseCorrespondence, 'correspondence')

    return denseCorrespondence

def computeDecomposition(image, stackSize):

    laplaceStack = []
    energyStack = []

    # normaliza a imagem de entrada para poder assumir valores negativos entre 0 e 1
    imageNorm = normalize(image)
  
    # tamanho do filtro gaussiano, o artigo indica para usar 5*2**i+1, sendo i a posição na pilha
    # como esse é o 0, fica 5*2+1 
    ksize = stackSize*stackSize

    # seguindo a formula, para a primeira entrada faz a imagem - gaussiano dela com desvio 2
    laplaceStack.append(imageNorm - cv2.GaussianBlur(imageNorm, (ksize,ksize), 2))

    for i in range(1, stackSize):        
        # para as proximas faz a gaussiana de 2**i - a gaussiana de 2**i+1   
        laplaceStack.append(cv2.GaussianBlur(imageNorm, (ksize,ksize), pow(2,i)) - cv2.GaussianBlur(imageNorm, (ksize,ksize), pow(2,i+1)))
        
    for i in range(stackSize):    
        # para o mapa de energia, calcula a saida com a laplace de mesmo indice ao quadrado, convolucionada
        # com gaussiana 2**i+1
        energyStack.append(cv2.GaussianBlur(np.square(laplaceStack[i]), (ksize,ksize), pow(2,i+1)))
    
    # o residuo é apenas uma imagem da convulação da entrada com filtro de desvio 2**tamanho da pilha
    residual = cv2.GaussianBlur(imageNorm, (ksize,ksize), pow(2,stackSize))
    
    # pode exibir ou salvar o resultado...
    #showStack(laplaceStack)
    writeStack(laplaceStack, 'laplace')
    writeStack(energyStack, 'energy')
    writeStack([residual], 'residual')
    #showStack(energyStack)

    # returna as duas pilhas e o residuo
    return laplaceStack, energyStack, residual

def robustTransfer(laplaceStack, energyStack, denseCorrespondence):

    # inicializa a saida
    outputStack = []
    stackSize = len(laplaceStack)

    # para cada posição na pilha
    for i in range(stackSize):

        # calcula o ganho de acordo com o artigo
        # a raiz quadrada da divisão da correspondencia do retrato com o mapa de energia da imagme de entrada
        # para evitar divisao 0/0 soma um valor infimo de 0,01**2
        gain = np.sqrt((denseCorrespondence[i])/((energyStack[i])+pow(0.01,2)))

        # segundo o artigo, o ganho deve ser truncado entre 0.9 e 2.8
        gain[gain > 2.8] = 2.8
        gain[gain < 0.9] = 0.9  

        ksize = stackSize*stackSize
        # por fim multiplica a laplaciana da entrada com uma gaussiana da imagem de ganho
        outputStack.append((laplaceStack[i])*cv2.GaussianBlur(gain, (ksize,ksize), 3*pow(2,i)))
    
    #showStack(outputStack)
    
    return outputStack

def sumStack(image, stack):

    # normaliza a imagem de entrada entre 0 e 1
    final_image = normalize(image)
    
    # para cada imagem na pilha, soma a entrada
    for img in stack:
        final_image += img
    
    # por ultimo normaliza a imagem de saida, 
    # ja que pode ficar com valores negativos ou maiores que 1
    final_image = (final_image - final_image.min()) / (final_image.max() - final_image.min())

    # e retorna ja em escala de 0 a 255
    return (final_image * 255).astype(np.uint8)

inputImage = ['farido.png']
portImage = ['gigachad.png']

for i in range(len(inputImage)):

    global batchName
    batchName = os.path.splitext(inputImage[i])[0]+'-'+os.path.splitext(portImage[i])[0]

    # carrega as imagens
    image_input = cv2.imread(inputImage[i])
    image_port = cv2.imread(portImage[i])

    # inicia o detector dos pontos da face
    detector = landmarksDetector()

    # carrega os pontos para a entrada e o retrato
    landmarks_input = detector.getLandmarks(image_input)
    landmarks_port = detector.getLandmarks(image_port)

    # verifica o numero da canais das imagens
    if len(image_input.shape)>2:
        channel_count = image_input.shape[2]
    else:
        channel_count = 1

    # inicializa a saida, vai ter 3 imagens se for colorido e 1 se for cinza
    output = []

    # para cada canal de cor, faz todo o processamento
    for channel in range(channel_count):
        
        # pega o canal correspondente de cada imagem
        input_gray = cv2.split(image_input)[channel]
        port_gray = cv2.split(image_port)[channel]

        # a funcao computeDecomposition retorna as pilhas do laplace, do mapa de energia e o residuo
        # usa o tamanho como 7 pois acima disso restam poucas altas frequencias
        input_laplace, input_energy, input_residual = computeDecomposition(input_gray, 7)
        _, port_energy, port_residual = computeDecomposition(port_gray, 7)

        # faz a correspondencia da image de entrada com o mapa de energia do retrato
        port_correspondence = denseCorrespondence(port_energy, landmarks_input, landmarks_port)

        # realiza a transferencia para cada posicao das pilhas
        robust_transfer = robustTransfer(input_laplace, input_energy, port_correspondence)
        
        # por fim faz a correspondencia da imagem de residuo do retrato
        robust_transfer.append(warpFace(port_residual, landmarks_port, landmarks_input))

        # desfaz as operações somando tudo da pilha na imagem de entrada
        # adicionando como um canal a imagem de saida
        output.append(sumStack(input_gray, robust_transfer))

    completeOutput = cv2.merge(output)

    writeStack([completeOutput], 'output')

    #image_merge = np.concatenate((input_gray, output), axis=1)
    #height, width = image_input.shape[0:2]

    #sift = cv2.SIFT_create()
    #kp = sift.detect(image_gray,None)
    #image=cv2.drawKeypoints(image_gray,kp,image)

    #cv2.imshow('input', image_input)
    #cv2.imshow('port', image_port)
    #cv2.imshow('style', completeOutput)
    #cv2.waitKey(0)