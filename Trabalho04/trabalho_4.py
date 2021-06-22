from tabulate import tabulate
from skimage.feature.texture import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from skimage.feature import local_binary_pattern
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
import sys

#Essa variável define se o código está sendo executado em notebook ou não,
#de modo que possa ser convertido em um script facilmente.

notebook = False

def compare_images(image1, image2):
  size = [10,4]
  #Conversão das Imagens
  image1_grayscale = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
  image2_grayscale = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

  radius = 3
  n_points = 8
  
  image1_lbp = local_binary_pattern(image1_grayscale, n_points, radius)
  image2_lbp = local_binary_pattern(image2_grayscale, n_points, radius)

  image1_lbp = (image1_lbp + np.abs(image1_lbp.min()))/(np.abs(image1_lbp.min()) + image1_lbp.max()) * 255
  image2_lbp = (image2_lbp + np.abs(image2_lbp.min()))/(np.abs(image2_lbp.min()) + image2_lbp.max()) * 255

  image1_lbp = image1_lbp.astype(np.uint8)
  image2_lbp = image2_lbp.astype(np.uint8)

  x_hist = np.arange(0,256)

  image1_hist = cv.calcHist(image1_lbp,[0],None,[256],[0,256])
  image2_hist = cv.calcHist(image2_lbp,[0],None,[256],[0,256])

  comp_chisqr = cv.compareHist(image1_hist, image2_hist, 1) 

  print("Distância qui-quadrado entre os histogramas 1 e 2: %d" %comp_chisqr)
  print("")

  image1_glcm = greycomatrix(image1_grayscale, distances=[1], angles=[np.pi/2])
  image2_glcm = greycomatrix(image2_grayscale, distances=[1], angles=[np.pi/2])

  image1_props = np.zeros([4])
  image2_props = np.zeros([4])


  image1_props[0] = greycoprops(image1_glcm, prop= 'ASM')
  image1_props[1] = greycoprops(image1_glcm, prop= 'contrast')
  image1_props[2] = shannon_entropy(image1_glcm)

  image2_props[0] = greycoprops(image2_glcm, prop= 'ASM')
  image2_props[1] = greycoprops(image2_glcm, prop= 'contrast')
  image2_props[2] = shannon_entropy(image2_glcm)


  print(tabulate([['Imagem 1', image1_props[0], image1_props[1], image1_props[2]], ['Imagem 2', image2_props[0], image2_props[1], image2_props[2]]], headers=['Image','ASM', 'Contrast', 'Entropy']))
  print("")

  #mostrando as imagens

  #Mostrando as imagens originais
  plt.figure(figsize=[10,15])
  plt.subplot(3,2,1)
  plt.imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
  plt.xticks([]), plt.yticks([])
  plt.title("Imagem 1")

  plt.subplot(3,2,2)
  plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
  plt.xticks([]), plt.yticks([])
  plt.title("Imagem 2")

  #Mostrando as imagens em escala de cinza
  plt.subplot(3,2,3)
  plt.imshow(cv.cvtColor(image1_grayscale, cv.COLOR_BGR2RGB))
  plt.xticks([]), plt.yticks([])
  plt.title("Escala de Cinza: Imagem 1")

  plt.subplot(3,2,4)
  plt.imshow(cv.cvtColor(image2_grayscale, cv.COLOR_BGR2RGB))
  plt.xticks([]), plt.yticks([])
  plt.title("Escala de Cinza: Imagem 2")

  #Mostrando o LBP das imagens
  plt.subplot(3,2,5)
  plt.imshow(image1_lbp, cmap="gray")
  plt.xticks([]), plt.yticks([])
  plt.title("LBP: Imagem 1")

  plt.subplot(3,2,6)
  plt.imshow(image2_lbp, cmap="gray")
  plt.xticks([]), plt.yticks([])
  plt.title("LBP: Imagem 2")

  plt.savefig('comparisons.png')
  if notebook:
    plt.show()

  plt.figure(figsize=size)
  plt.subplot(1,2,1)
  plt.title("Histograma LBP: Imagem 1")
  plt.bar(x_hist, image1_hist.reshape([256]))

  plt.subplot(1,2,2)
  plt.title("Histograma LBP: Imagem 2")
  plt.bar(x_hist, image2_hist.reshape([256]))
  plt.savefig('comparisons_hist.png')
  if notebook:
    plt.show()

  return 0



arquivo1 = sys.argv[1]
arquivo2 = sys.argv[2]
img1 = cv.imread(arquivo1) 
img2 = cv.imread(arquivo2) 


compare_images (img1,img2)
