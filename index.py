"""                   DEEPFLOW PROJECT : ANOMALY CLASSIFIER PIPELINE                             """

import numpy as np
import cv2
import matplotlib.pyplot as plt


""" -------------------------------LOADING AND COPYING THE IMAGE  --------------------------------------"""
 
original_image = cv2.imread( "C:\\Users\\pierr\\Desktop\\os2.png" )
img = original_image


""" ------------------------------------------CONTRAST ---------------------------------------------------"""

def contrast_step( img ):
    
    alpha = 1.07 # Contrast control (1.0-3.0)
    beta = 50 # Brightness control (0-100)
    manual_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta )
    
    figure_size = 15
    plt.figure(figsize = (figure_size,figure_size) )
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(manual_contrast)
    plt.title('manual contrast' ), plt.xticks([]), plt.yticks([])
    plt.show()

    return( manual_contrast )

contrast_step( original_image )

""" ----------------------------------CLUSTERING : K-MEANS -----------------------------------------"""

def kmeans_step( img ):
    
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0 )
    
    K = 8
    attempts=10
    ret, label, center = cv2.kmeans( vectorized, K, None, criteria, attempts,cv2.KMEANS_PP_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    
    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(result_image)
    plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
    plt.show()
    
    return( result_image )

kmeans_step ( original_image )


""" ------------------------------------EDGE DETECTION -----------------------------------------------------"""

def edges_step(img):
    
    edges = cv2.Canny(img , 10, 50)
    
    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    return(edges)

edges_step( original_image )


""" ----------------- FEATURES EXTRACTION WITH GLCM --------------------------------"""


from skimage.feature import greycomatrix, greycoprops

def GLCM_features_step(img):
    """def convert_glcm(glcm):
        cpy_glcm=[]
        for kol in range (0, 256):
            cpy_glcm.append ([])
            for kal in range (0,256): 
                cpy_glcm[kol].append (glcm[kol][kal][0][0])
        return np.array(cpy_glcm)
    #Ouverture des images 
    imggrass = cv2.imread("C:\\Users\\pierr\\Desktop\\os2.png")
    imggrass =cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    
    #Calcul des matrices de co-occurence des différentes images
    glcmgrass = greycomatrix(imggrass[:,:,0], distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=False)
    
    #Normalisation des matrices de co-occurence des images afin de pouvoir les afficher
    img_glcmgrass=cv2.normalize(np.array(glcmgrass), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    
    #Conversion des matrices de co-occurences des images afin de pouvoir les afficher
    img_glcmgrass=convert_glcm(img_glcmgrass)
    
    #Affichage des matrices de co-occurences des différentes images
    cv2.imshow("glcm grass", img_glcmgrass)
    
    #Calcul à partir de la dissimilarité et de la corrélation de chacune des matrices de co-occurences
    dissimilarite_grass=greycoprops(glcmgrass, 'dissimilarity')[0, 0]
    correlation_grass=greycoprops(glcmgrass, 'correlation')[0, 0]
    
    #insertion  de la dissimilarité et de la corrélation dans un tableau afin de pouvoir les afficher avec la fonction scatter de matplolib
    tableau_grass=[[dissimilarite_grass],[correlation_grass]]
    
    
    #Affichage de la corrélation et de la dissimilarité pour les trois images
    plt.scatter(tableau_grass[0],tableau_grass[1],label='Grass',color='r')
    
    plt.xlabel('Dissimilarité')
    plt.ylabel('Corrélation')
    plt.title('Exemples de mesures de la matrice de co-occurence de différentes textures')
    plt.legend()
    plt.show()"""
    df = 666
    
    return df

"""########################################################################################"""
"""                          FINAL PREPROCESSING                                 """
"""########################################################################################"""

def preprocess(img):
    
    #img = contrast_step(img)
    #img = kmeans_step(img)
    img = edges_step(img)
    df = GLCM_features_step(img)
    
    print("the preprocessing procedure is fully completed")
    
    return df

preprocess ( original_image )


"""classif with decision trees, SVM, naive bayes"""

preprocess X_train
preprocess Xtest

accuracy score


""" IMPROVEMENTS """

try to find the corner detection : harris corner detection