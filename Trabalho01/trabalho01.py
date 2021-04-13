# MO443 - Introdução ao Processamento de imagem digital | Trabalho 01

#Importando as bibliotecas que são usadas ao longo do trabalho
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import time

#índices
r = 2
g = 1
b = 0


#Carregando a imagem
def processaColoridoSepia(arquivo):
    imagem = cv.imread(arquivo)
    imagemProcessada = np.ones_like(imagem)
    
    #Processando a figura, usando composições das três cores. É interessante observar que o operador +, para ndarray, é o mesmo que np.add().
    imagemProcessada[:,:,r] = np.multiply(imagem[:,:,r], 0.0393) + np.multiply(imagem[:,:,g], 0.769)
    + np.multiply(imagem[:,:,b], 0.189)
    imagemProcessada[:,:,g] = np.multiply(imagem[:,:,r], 0.0349) + np.multiply(imagem[:,:,g], 0.686)
    + np.multiply(imagem[:,:,b], 0.168)
    imagemProcessada[:,:,b] = np.multiply(imagem[:,:,r], 0.0272) + np.multiply(imagem[:,:,g], 0.534)
    + np.multiply(imagem[:,:,b], 0.131)

    #Definindo um limite superior para os valores da imagem processada
    imagemProcessada[imagemProcessada > 255] = 255
	
    #Salvando a imagem processada
    cv.imwrite("sepia_" + arquivo, imagemProcessada)

def processaColoridoCinza(arquivo):
    #Carregando a imagem e criando uma imagem de mesma forma porém com um só canal.
    #imagem = cv.imread("monalisa.png")
    #imagem = np.zeros([128,128,3])
    imagem = cv.imread(arquivo)
    imagemProcessada = np.zeros_like(imagem)

    #Convertendo a imagem para escala de cinza
    imagemProcessada[:,:,r] = (np.multiply(imagem[:,:,r], 0.2989) + np.multiply(imagem[:,:,g], 0.5870) + np.multiply(imagem[:,:,b], 0.1140))
    imagemProcessada[:,:,g] = (np.multiply(imagem[:,:,r], 0.2989) + np.multiply(imagem[:,:,g], 0.5870) + np.multiply(imagem[:,:,b], 0.1140))
    imagemProcessada[:,:,b] = (np.multiply(imagem[:,:,r], 0.2989) + np.multiply(imagem[:,:,g], 0.5870) + np.multiply(imagem[:,:,b], 0.1140))

    #Salvando a imagem processada
    cv.imwrite("escalaCinza_" + arquivo, imagemProcessada)

def convoluciona(imagem, kernel):
    #Ajustando o frame em volta da imagem
    #O frame aqui é simplesmente preenchido com o valor médio da imagem

    tamanhoImagem = [imagem.shape[0],imagem.shape[1]]
    tamanhoKernel = [kernel.shape[0],kernel.shape[1]]
    passoKernel = [int(tamanhoKernel[0]/2),int(tamanhoKernel[1]/2)]
    imagemFrame = np.ones([imagem.shape[0] + 2 * int(kernel.shape[0]/2), imagem.shape[0] + 2 * int(kernel.shape[0]/2)])* int(np.mean(imagem))
    imagemFrame[int(tamanhoKernel[0]/2):-int(tamanhoKernel[0]/2), int(tamanhoKernel[1]/2):-int(tamanhoKernel[0]/2)] = imagem[:,:]

    #Desmontando a imagem em blocos linearizados
    blocosImagem = np.zeros([tamanhoImagem[0] * tamanhoImagem[1],tamanhoKernel[0] * tamanhoKernel[1]])
    a = 0
    for i in range(passoKernel[0], imagemFrame.shape[0] - passoKernel[0]):
        for j in range(passoKernel[1], imagemFrame.shape[1] - passoKernel[1]):
            bloco = imagemFrame[i - passoKernel[0]:i + passoKernel[0] +1, j - passoKernel[1]:  j + passoKernel[1]+1]
            blocosImagem[a,:] = np.reshape(bloco,[1,kernel.shape[0] * kernel.shape[1]])
            a = a + 1;

    #Aplicando a convolução
    kernelLinear = np.ravel(kernel)
    imagemProcessada = np.matmul(kernelLinear, blocosImagem.T)

    #Remontando a imagem
    resultado  = np.reshape(imagemProcessada, [imagem.shape[0], imagem.shape[1]])
    
    #Normalizando a imagem
    resultado = (resultado + np.abs(resultado.min()))/resultado.max() *255

    return resultado


def processaCinza(arquivo):
    imagem = cv.imread(arquivo, 0)
    
    #Iniciando as matrizes dos filtros
    h1 = np.array([[-1,  0,  1], [-2, 0, 2], [-1, 0, 1]])

    h2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    h12 = np.sqrt(np.add(np.power(h1,2), np.power(h2,2)))

    h3 = np.full([3, 3],-1)
    h3[1,1] = 8;

    h4 = np.ones([3,3])/9

    h5 = np.full([3,3], -1)
    np.fill_diagonal(h5,2)

    h6 = h5
    h5 = np.fliplr(h5)

    h7 = np.zeros([3,3])
    h7[0,2] = 1
    h7[2,0] = -1

    h8 = np.array([[0, 0, -1, 0, 0],[0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])

    h9 = np.array([1, 4, 6, 4, 1])
    aux = np.full([5,5],1)
    aux = np.multiply(aux,h9)
    h9 = np.multiply(aux.T,h9)/256

    #Aplicando as convoluções
    print("Convoluindo com H1")
    imagem_h1  = convoluciona(imagem, h1)

    print("Convoluindo com H2")
    imagem_h2  = convoluciona(imagem, h2)

    print("Convoluindo com H12")
    imagem_h12 = convoluciona(imagem, h12)

    print("Convoluindo com H3")
    imagem_h3  = convoluciona(imagem, h3)

    print("Convoluindo com H4")
    imagem_h4  = convoluciona(imagem, h4)

    print("Convoluindo com H5")
    imagem_h5  = convoluciona(imagem, h5)

    print("Convoluindo com H6")
    imagem_h6  = convoluciona(imagem, h6)

    print("Convoluindo com H7")
    imagem_h7  = convoluciona(imagem, h7)

    print("Convoluindo com H8")
    imagem_h8  = convoluciona(imagem, h8)

    print("Convoluindo com H9")
    imagem_h9  = convoluciona(imagem, h9)

    #Salvando as imagens
    print("Salvando novas imagens")
    cv.imwrite("h1_" + arquivo, imagem_h1)
    cv.imwrite("h2_" + arquivo, imagem_h2)
    cv.imwrite("h12" + arquivo, imagem_h12)
    cv.imwrite("h3_" + arquivo, imagem_h3)
    cv.imwrite("h4_" + arquivo, imagem_h4)
    cv.imwrite("h5_" + arquivo, imagem_h5)
    cv.imwrite("h6_" + arquivo, imagem_h6)
    cv.imwrite("h7_" + arquivo, imagem_h7)
    cv.imwrite("h8_" + arquivo, imagem_h8)
    cv.imwrite("h9_" + arquivo, imagem_h9)


try:
    arg = sys.argv[1]
    arquivo = sys.argv[2]
    
    if(arg == "-g"):
        print("Processando a imagem para escala de cinza")
        processaCinza(arquivo)
        print("Processamento concluído.")

    elif(arg == "-c"):
        print("Processando a imagem colorida")
        processaColoridoSepia(arquivo)
        processaColoridoCinza(arquivo)
        print("Processamento concluído.")
    
except:
    print("Sintaxe: trabalho01.py -c/g filename")



    

