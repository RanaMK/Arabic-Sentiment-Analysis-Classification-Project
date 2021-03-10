# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:21:30 2017
@author: PAVILION G4
"""
import codecs
import numpy as np
import pandas as pd
import re
from LoadDataset import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from qalsadi import analex
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import Perceptron
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn import metrics
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.ensemble.forest import RandomForestClassifier
from numpy.lib.scimath import sqrt
from numpy.ma.core import floor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcess
from sklearn import svm
from sklearn import preprocessing
from pickle import FALSE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import TruncatedSVD

# tokenizer
an = analex.analex()
tokenizer = an.text_tokenize

tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 1))

############# Balanced dataset ###########
d_train = np.concatenate((balanced_train_x, balanced_valid_x))
X_train = tf_fg.fit_transform(d_train)
Y_train = np.concatenate((balanced_train_y, balanced_valid_y))
X_test = tf_fg.transform(balanced_test_x)
Y_test = balanced_test_y

############# Unbalanced dataset ######
d_train = np.concatenate((unbalanced_train_x, unbalanced_valid_x))
X_train = tf_fg.fit_transform(d_train)
Y_train = np.concatenate((unbalanced_train_y, unbalanced_valid_y))
X_test = tf_fg.transform(unbalanced_test_x)
Y_test = unbalanced_test_y

########################### KNN   ##########################--Code from course
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

###########################   SVM ##############################--Code from course
# Fitting SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

######################  Logistic Regression #########################--Code from course
# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

######################  Passive Aggressive ###########################--Code from ASTD
classifier = PassiveAggressiveClassifier(n_iter=100)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

###################### Perceptron  ###################################--Code from ASTD
classifier = Perceptron(n_iter=100)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

######################  bnb ###########################################--Code from ASTD
classifier = BernoulliNB(binarize=0.5)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

###################### mnb ############################################--Code from ASTD
classifier = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

################# sgd ###############################################--Code from ASTD
classifier = SGDClassifier(loss="hinge", penalty="l2")
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)



#########################  ANN ##########################---to be checked and take the lesson again 
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder(sparse=False) # Key here is sparse=False!
#y_categorical = enc.fit_transform(Y_train.reshape((Y_train.shape[0]),1))
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y_train = LabelEncoder()
Y_train = labelencoder_Y_train.fit_transform(Y_train)
#Y_train = keras.utils.to_categorical(Y_train, 4)
onehotencoder = OneHotEncoder(categorical_features = [0])
Y_train = Y_train.reshape((-1, 1))
Y_train = onehotencoder.fit_transform(Y_train).toarray()
labelencoder_Y_test = LabelEncoder()
Y_test = labelencoder_Y_test.fit_transform(Y_test)
onehotencoder = OneHotEncoder(categorical_features = [0])
Y_test = Y_test.reshape((-1, 1))
Y_test = onehotencoder.fit_transform(Y_test).toarray()


# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', 
                     input_dim = 33350))  ##16455 for balanced and 33350 for unbalanced
## Adding the second hidden layer
#classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'softmax'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train.toarray(), Y_train, batch_size = 20, nb_epoch = 100, verbose=1)

#######Classifer Evaluation ###########
score = classifier.evaluate(X_test.toarray(), Y_test,
                       batch_size=20, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Predicting the Test set results
y_pred = classifier.predict(X_test.toarray())
#y_pred = (y_pred > 0.5)

for i in range(0,len(y_pred)):
    #len(y_pred)):
    max_y_pred = max(y_pred[i, :])
    for j in range (0,4):
        if y_pred[i, j] == max_y_pred:
            y_pred[i, j] = 1
        else:
            y_pred[i, j] =0    
#        if j == 0:
#            max_y_pred = y_pred[i, j]
#        if y_pred[i, j] > max_y_pred:
#            max_y_pred = y_pred[i, j]
#            print("i", i)
#            print("j", j)
#            print("y_pred", y_pred[i, j])
#            y_pred[i,j] = 1
#            y_pred[i,j-1] = 0  
#            print("max", max_y_pred)
#        else:
#            print("else", y_pred[i, j])
#        if j == 3 and y_pred[i, j] < max_y_pred:
#            y_pred[i, j] = 0
#            print("last iter", max_y_pred)
    i = i +1

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


##################### Convlutional Neural Networks #####################
#Building the CNN
from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y_train = LabelEncoder()
Y_train = labelencoder_Y_train.fit_transform(Y_train)
#Y_train = keras.utils.to_categorical(Y_train, 4)
onehotencoder = OneHotEncoder(categorical_features = [0])
Y_train = Y_train.reshape((-1, 1))
Y_train = onehotencoder.fit_transform(Y_train).toarray()
labelencoder_Y_test = LabelEncoder()
Y_test = labelencoder_Y_test.fit_transform(Y_test)
onehotencoder = OneHotEncoder(categorical_features = [0])
Y_test = Y_test.reshape((-1, 1))
Y_test = onehotencoder.fit_transform(Y_test).toarray()
## Initialising the CNN
#classifier = Sequential()
## Step 1 - Convolution
##classifier.add(Convolution2D(number_of_filters, number_ofrows_of each filter, 
##number_of columns_of each filter, input_shape = (2560, 16455, 1), activation = 'relu'))
#classifier.add(Convolution1D(32, 4, input_shape = 16455, activation = 'relu'))
## Step 2 - Pooling
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
## Adding a second convolutional layer
##classifier.add(Convolution1D(32, 16455, activation = 'relu'))
##classifier.add(MaxPooling2D(pool_size = (2, 2)))
##To avoid overfitting
##classifier.add(Dropout(0.25))
## Step 3 - Flattening
#classifier.add(Flatten())
## Step 4 - Full connection
#classifier.add(Dense(output_dim = 128, activation = 'relu'))
#classifier.add(Dense(output_dim = 4, activation = 'softmax'))
## Compiling the CNN
#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#classifier.fit(X_train, Y_train, batch_size=20, nb_epoch=100, verbose=1)
##fit_generator(X_train, samples_per_epoch = 100, nb_epoch = 20, validation_data = [X test,Y_test], nb_val_samples = 2000)
#accuracy = classifier.evaluate(X_test, Y_test, verbose=0)

#Ref: https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
classifier = Sequential()
classifier.add(Embedding(100, 32, input_length=16455))
classifier.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
classifier.add(MaxPooling1D(pool_size=2))

classifier.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
classifier.add(MaxPooling1D(pool_size=2))
#classifier.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
#classifier.add(MaxPooling1D(pool_size=2))

classifier.add(Flatten())
classifier.add(Dense(250, activation='relu'))
classifier.add(Dense(4, activation='softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(classifier.summary())
classifier.fit(X_train.toarray(), Y_train, validation_data=(X_test.toarray(), Y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = classifier.evaluate(X_test.toarray(), Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
