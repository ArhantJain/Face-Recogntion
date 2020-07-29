import os
#import cv2
import pickle
import numpy as np
import pdb
import requests
from collections import defaultdict
import random 
import time
import sys
import math
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FastICA as ICA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as CART
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import *
from sklearn.neural_network import MLPClassifier
from functools import wraps 
from time import time as _timenow 
from sys import stderr
import tensorflow as tf




def load_cifar():
    
    trn_data, trn_labels, tst_data, tst_labels = [], [], [], []
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
            print(np.shape(data))
        return data
    
    

    for i in trange(5):
        #print('Kai')
        #print(i)
        #batchName = './data/data_batch_{0}'.format(i + 1)
        batchName = './data/data_batch_{0}'.format(i + 1)
        #print(batchName)
        unpickled = unpickle(batchName)
        #print(unpickled)
        trn_data.extend(unpickled['data'])
        trn_labels.extend(unpickled['labels'])
    unpickled = unpickle('./data/test_batch')
    tst_data.extend(unpickled['data'])
    tst_labels.extend(unpickled['labels'])
    return trn_data, trn_labels, tst_data, tst_labels


def image_prep(image):
    

    #scaler = 




    #return tf.image.per_image_standardization(tf.image.rgb_to_grayscale(np.reshape(image,(32,32,3))))
    nc = len(image)

    #mu = 0
    #sigma = 0

    #processed_image = np.zeros(nc)



    #for i in trange(nc):
    #   mu = mu + image[i]

    #for i in trange(nc):
    #   processed_image[i] = image[i] - mu


    #for i in trange(nc):
    #   sigma = sigma + processed_image[i]*processed_image[i]


    #sigma = math.sqrt(sigma)

    #for i in trange(nc):
    #   processed_image[i] = processed_image[i]/sigma

    #return processed_image


def mu_sigma(matrix,red):
    #noOfsamplesXnoOffeatures
    scaler = StandardScaler()
    #scaler.fit(matrix)

    scaler.fit(matrix)
    matrix = scaler.transform(matrix)
    red = scaler.transform(red)
    return matrix , red



    nr , nc = np.shape(matrix)
    


    mu = np.zeros( nc, dtype = int) 
    sigma = np.zeros(nc, dtype = int)


    print(nr)
    nmatrix = np.zeros([nr,nc], dtype = int)




    for j in trange(nc):
        for i in trange(nr):
            #print(i,j)
            mu[j] = mu[j] + matrix[i][j]

        mu[j] = mu[j]*(1/nr)

        for i in trange(nr):
            nmatrix[i][j] = matrix[i][j] - mu[j]

    for j in trange(nc):
        for i in trange(nr):
            sigma[j] = sigma[j] + nmatrix[i][j]*nmatrix[i][j]

        sigma[j] = math.sqrt(sigma[j]*(1/nr))

        for i in trange(nr):
            nmatrix[i][j] = nmatrix[i][j]/sigma[j]


    return mu, sigma, nmatrix



#kwargs contains method-PCA LDA or ICA, data - the samples , y - labels , n_components - No of componenets  (hyperparameter)
def reduce_dim(data,labels,n_components,**kwargs):
    ''' performs dimensionality reduction'''
    if kwargs['method'] == 'pca':
        

        matrix = data
        pca = PCA(n_components=n_components)
        pca.fit(matrix)
        #return pca.fit_transform(matrix)
        #pass
        return pca.transform(matrix)
        
    if kwargs['method'] == 'lda':
        
        label = labels
        matrix = data
        lda = LDA(n_components = n_components)
        lda.fit(matrix,label)
        LDA(n_components= n_components, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)
        #return lda.fit_transform(matrix,label)
        return lda.transform(matrix)


        pass    


    if kwargs['method'] == 'ica':
        
        matrix = data
        ica = ICA(n_components = n_components,random_state = 0)
        return ica.fit_transform(matrix)


        
#train the classifier - X is noOfsamplesXnoOffeatures ; y is noOfsamples ,method - type of classifier
def classify(data,labels,**kwargs):
    ''' trains a classifier by taking input features
        and their respective targets and returns the trained model'''
    if kwargs['method'] == 'SVM':

        clf = LinearSVC(random_state=0, tol=1e-5)
        matrix = data
        label = labels
        return clf.fit(matrix,label)

        pass


    if kwargs['method'] == 'CART':
        clf = DecisionTreeClassifier(random_state=0)
        matrix = data
        label = labels
        return clf.fit(matrix,label)


    if kwargs['method'] == 'KSVM':
        matrix = data
        label = labels
        clf = SVC(kernel='rbf')
        return clf.fit(matrix,label)

    if kwargs['method'] == 'MLP':
        matrix = data
        label = labels
        clf = MLPClassifier(activation= 'relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(50,50),momentum = 0.9, random_state=1,max_iter=200,nesterovs_momentum = True,learning_rate_init = 0.001,learning_rate = 'constant')
        return clf.fit(matrix,label)

        
def evaluate(target, predicted):
    f1 = f1_score(target, predicted, average='micro')
    acc = accuracy_score(target, predicted)
    return f1, acc

def test(data,labels,clf,**kwargs):

    #classifier = kwargs['clf']
    classifier = clf

    target = labels

    test = data

    predicted = classifier.predict(test)

    return evaluate(target, predicted)
    '''takes test data and trained classifier model,
    performs classification and prints accuracy and f1-score'''
    

def main():
    trn_data, trn_labels, tst_data, tst_labels = load_cifar()
    #trn_datas = list(map(lambda x : tf.image.rgb_to_grayscale(np.reshape(x, (32,32,3))),trn_data))
    trn_data, tst_data = mu_sigma(trn_data,tst_data)



    #trn_data, tst_data = list(map(image_prep, trn_data)), list(map(image_prep, tst_data))
    X_train, X_val, y_train, y_val = train_test_split(trn_data, trn_labels,test_size = 0.20) 
    ''' perform dimesioality reduction/feature extraction and classify the features into one of 10 classses
        print accuracy and f1-score.
        '''
    #mu, sigma, nmatr3072= mu_sigma(trn_data)

    print(np.shape(X_train))
    print(np.shape(X_val))
    print(np.shape(y_train))
    print(np.shape(y_val))

    #sys.exit()






    #rPCA = reduce_dim(X_train,y_train,100, method ='pca')    
    #rLDA = reduce_dim(X_train,y_train,3072,method ='lda')
    #rICA = reduce_dim(X_train,y_train,256,method='ica')
    rRAW = X_train

    rLabel = y_train

    #print(np.shape(rPCA))
    #print(np.shape(rLDA))
    #print(np.shape(rRAW))
    #print(np.shape(rICA))

    #oPCA = reduce_dim(tst_data,tst_labels,100,method = 'pca')
    #oLDA = reduce_dim(tst_data,tst_labels,256,method = 'lda')
    oRAW = tst_data

    oLabel = tst_labels

    #vPCA = reduce_dim(y_train,y_val,100,method = 'pca')

    #vLabel = y_val



    #sys.exit()




    #cSVMp = classify(rPCA,rLabel,method = 'SVM')
    #cSVMl = classify(rLDA,rLabel,method = 'SVM')
    #cSVMi = classify(data=rICA,labels = rLabel,method = 'SVM')
    #cSVMr = classify(rRAW,rLabel,method = 'SVM')


    #cMLPp = classify(rPCA,rLabel,method = 'MLP')
    #cMLPl = classify(rLDA,rLabel,method = 'MLP')
    #cMLPi = classify(data=rICA,labels = rLabel,method = 'MLP')
    cMLPr = classify(rRAW, rLabel,method = 'KSVM')

    #cKSVMp = classify(rPCA, rLabel,method = 'KSVM')
    #cKSVMl = classify(rLDA, rLabel,method = 'KSVM')
    #cKSVMi = classify(data=rICA,labels = rLabel,method = 'KSVM')
    #cKSVMr = classify(rRAW, rLabel,method = 'KSVM')

    #cCARTp = classify(rPCA, rLabel,method = 'CART')
    #cCARTl = classify(rLDA, rLabel,method = 'CART')
    #cCARTi = classify(data=rICA,labels = rLabel,method = 'CART')
    #cCARTr = classify(rRAW, rLabel,method = 'CART')


    F, A = test(oRAW,oLabel,cMLPr)

    print(F)
    print(A)

    sys.exit()



    #print(np.shape(X_train))
    #print(np.shape(X_val))
    #print(np.shape(y_train))
    #print(np.shape(y_val))





    #print(X_val)

    #print(trn_data)
    #print(trn_labels)



    


    print('Val - F1 score: {}\n Accuracy: {}'.format(f_score, accuracy_))    



if __name__ == '__main__':
    main()





