#!/usr/bin/env python
# coding: utf-8

# # MO443 - Introdução ao Processamento de imagem digital | Trabalho 01

# In[2]:


#Importando as bibliotecas que são usadas ao longo do trabalho
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys

#índices
r = 2
g = 1
b = 0


# ## Exercício 1.1a:
# 
# Este primeiro exercício está pautado na manipulação dos três canais de cor de uma imagem. Para a primeira letra, a imagem resultante tem como cada um dos seus canais uma associação dos canais da imagem original, poderados por diferentes pesos definidos previamente. Isso pode ser solucionado facilmente por um slicing da imagem original recebendo todos os pixels de um certo canal e multiplicando-os com uma multiplicação elemento-a-elemento, isso segue então por uma soma do mesmo feito para os outros dois canais.
# 
# 

# In[3]:


#Carregando a imagem
#imagem = np.zeros([128,128,3])
def processaColoridoSepia(arquivo):
    imagem = cv.imread(arquivo)
    imagemProcessada = np.ones_like(imagem)

    cv.imshow('image',imagem)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #Processando a figura, usando composições das três cores. É interessante observar que o operador +, para ndarray, é o mesmo que np.add().
    imagemProcessada[:,:,r] = np.multiply(imagem[:,:,r], 0.0393) + np.multiply(imagem[:,:,g], 0.769) + np.multiply(imagem[:,:,b], 0.189)
    imagemProcessada[:,:,g] = np.multiply(imagem[:,:,r], 0.0349) + np.multiply(imagem[:,:,g], 0.686) + np.multiply(imagem[:,:,b], 0.168)
    imagemProcessada[:,:,b] = np.multiply(imagem[:,:,r], 0.0272) + np.multiply(imagem[:,:,g], 0.534) + np.multiply(imagem[:,:,b], 0.131)

    #Definindo um limite superior para os valores da imagem processada
    imagemProcessada[imagemProcessada > 255] = 255

    #Mostrando a imagem
    plt.figure(figsize=(10,10))


    plt.subplot(1,2,1)
    plt.title("Imagem Original")
    plt.axis("off")
    plt.imshow(imagem)


    plt.subplot(1,2,2)
    plt.title("Imagem Processada")
    plt.axis("off")
    plt.imshow(imagemProcessada)

    plt.show()

    #Salvando a imagem processada
    cv.imwrite("sepia_" + arquivo, imagemProcessada)


# ## Exercício 1.1b:

# In[5]:


def processaColoridoSepia(arquivo):
    #Carregando a imagem e criando uma imagem de mesma forma porém com um só canal.
    #imagem = cv.imread("monalisa.png")
    #imagem = np.zeros([128,128,3])
    imagem = cv.imread(arquivo)
    imagemProcessada = np.zeros_like(imagem)

    #Convertendo a imagem para escala de cinza
    imagemProcessada[:,:,r] = (np.multiply(imagem[:,:,r], 0.2989) + np.multiply(imagem[:,:,g], 0.5870) + np.multiply(imagem[:,:,b], 0.1140))
    imagemProcessada[:,:,g] = (np.multiply(imagem[:,:,r], 0.2989) + np.multiply(imagem[:,:,g], 0.5870) + np.multiply(imagem[:,:,b], 0.1140))
    imagemProcessada[:,:,b] = (np.multiply(imagem[:,:,r], 0.2989) + np.multiply(imagem[:,:,g], 0.5870) + np.multiply(imagem[:,:,b], 0.1140))

    #Mostrando a imagem
    plt.figure(figsize=(10,10))

    plt.subplot(1,2,1)
    plt.title("Imagem Original")
    plt.axis("off")
    plt.imshow(imagem)

    plt.subplot(1,2,2)
    plt.title("Imagem Processada")
    plt.axis("off")
    plt.imshow(imagemProcessada)

    plt.show()
    
    #Salvando a imagem processada
    cv.imwrite("escalaCinza_" + arquivo, imagemProcessada)


# ## Exercício 1.2:

# In[7]:


def convoluciona(imagem, kernel):
    #Ajustando o frame em volta da imagem
    #O frame aqui é simplesmente preenchido com o valor médio da imagem
    tamanhoImagem = [imagem.shape[0],imagem.shape[1]]
    tamanhoKernel = [kernel.shape[0],kernel.shape[1]]
    passoKernel = [int(tamanhoKernel[0]/2),int(tamanhoKernel[1]/2)]
    imagemFrame = np.ones([imagem.shape[0] + 2 * int(kernel.shape[0]/2), imagem.shape[0] + 2 * int(kernel.shape[0]/2)])* int(np.mean(imagem))
    imagemFrame[int(tamanhoKernel[0]/2):-int(tamanhoKernel[0]/2), int(tamanhoKernel[1]/2):-int(tamanhoKernel[0]/2)] = imagem[:,:,0]
    
    #Desmontando a imagem em blocos linearizados
    blocosImagem = np.empty([0,tamanhoKernel[0] * tamanhoKernel[1]],dtype = int)
    
    for i in range(passoKernel[0], imagemFrame.shape[0] - passoKernel[0]):
        for j in range(passoKernel[1], imagemFrame.shape[1] - passoKernel[1]):
            bloco = imagemFrame[i - passoKernel[0]:i + passoKernel[0] +1, j - passoKernel[1]:  j + passoKernel[1]+1]
            blocosImagem = np.append(blocosImagem, np.reshape(bloco,[1,kernel.shape[0] * kernel.shape[1]]), 0)
            
            
    #Aplicando a convolução
    kernelLinear = np.ravel(kernel)
    imagemProcessada = np.matmul(kernelLinear, blocosImagem.T)
    
    #Remontando a imagem
    resultado  = np.reshape(imagemProcessada, [imagem.shape[0], imagem.shape[1]])
    
    return resultado


# In[20]:


def processaCinza(arquivo):
    imagem = cv.imread(arquivo)
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


    imagem_h1  = convoluciona(imagem, h1)
    imagem_h2  = convoluciona(imagem, h2)
    imagem_h12 = convoluciona(imagem, h12)
    imagem_h3  = convoluciona(imagem, h3)
    imagem_h4  = convoluciona(imagem, h4)
    imagem_h5  = convoluciona(imagem, h5)
    imagem_h6  = convoluciona(imagem, h6)
    imagem_h7  = convoluciona(imagem, h7)
    imagem_h8  = convoluciona(imagem, h8)
    imagem_h9  = convoluciona(imagem, h9)


    plt.figure(figsize=(20,10))

    plt.subplot(2,5,1)
    plt.title("Kernel H1")
    plt.axis("off")
    plt.imshow(imagem_h1, cmap='gray')

    plt.subplot(2,5,2)
    plt.title("Kernel H2")
    plt.axis("off")
    plt.imshow(imagem_h2, cmap='gray')

    plt.subplot(2,5,3)
    plt.title("Kernel H1+H2")
    plt.axis("off")
    plt.imshow(imagem_h12, cmap='gray')

    plt.subplot(2,5,4)
    plt.title("Kernel H3")
    plt.axis("off")
    plt.imshow(imagem_h3, cmap='gray')

    plt.subplot(2,5,5)
    plt.title("Kernel H4")
    plt.axis("off")
    plt.imshow(imagem_h4, cmap='gray')

    plt.subplot(2,5,6)
    plt.title("Kernel H5")
    plt.axis("off")
    plt.imshow(imagem_h5, cmap='gray')

    plt.subplot(2,5,7)
    plt.title("Kernel H6")
    plt.axis("off")
    plt.imshow(imagem_h6, cmap='gray')

    plt.subplot(2,5,8)
    plt.title("Kernel H7")
    plt.axis("off")
    plt.imshow(imagem_h7, cmap='gray')

    plt.subplot(2,5,9)
    plt.title("Kernel H8")
    plt.axis("off")
    plt.imshow(imagem_h8, cmap='gray')

    plt.subplot(2,5,10)
    plt.title("Kernel H9")
    plt.axis("off")
    plt.imshow(imagem_h9, cmap='gray')

    plt.show()

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


# In[ ]:


arg = sys.argv[1]
arquivo = sys.argv[2]

if(arg == "-g"):
    print("Processando a imagem em escala de cinza: ")
    processaCinza(arquivo)
    
elif(arg == "-c"):
    print("Processando a imagem colorida: ")
    processaColoridoSepia(arquivo)
    processaColoridoBW(arquivo)
    
    
else:
    print("Argumento inválido")

