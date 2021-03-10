# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:24:21 2017

@author: Rana Mahmoud
"""
import re
import nltk
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
import nltk
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
import numpy as np
from sklearn.naive_bayes import ComplementNB
        
class Classifiers:
    def __init__(self):
        self.classifiers = ('KNN', 'SVM','Logistic_Regression','Passive_Aggressive','Perceptron',
                       'bnb','mnb','sgd','ANN','CNN')
    
    def KNN(X_train, Y_train, X_test, Y_test): 
        ########################### KNN   ##########################--Code from course
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, Y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, y_pred)
        if len(cm[0]) == 2:
            total_correct_predictions = cm[0,0]+cm[1,1]
        elif len(cm[0]) == 3:
            total_correct_predictions = cm[0,0]+cm[1,1]+cm[2,2]
        total_predictions_made = np.sum(cm)
        accuracy = total_correct_predictions / total_predictions_made * 100
        
        return accuracy

    def SVM(X_train, Y_train, X_test, Y_test):
        ###########################   SVM ##############################--Code from course
        # Fitting SVM to the Training set
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, Y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, y_pred)
        if len(cm[0]) == 2:
            total_correct_predictions = cm[0,0]+cm[1,1]
        elif len(cm[0]) == 3:
            total_correct_predictions = cm[0,0]+cm[1,1]+cm[2,2]
        total_predictions_made = np.sum(cm)
        accuracy = total_correct_predictions / total_predictions_made * 100
        
        return accuracy

    def Logistic_Regression(X_train, Y_train, X_test, Y_test):
        ######################  Logistic Regression #########################--Code from course
        # Fitting Logistic Regression to the Training set
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, Y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, y_pred)
        if len(cm[0]) == 2:
            total_correct_predictions = cm[0,0]+cm[1,1]
        elif len(cm[0]) == 3:
            total_correct_predictions = cm[0,0]+cm[1,1]+cm[2,2]
        total_predictions_made = np.sum(cm)
        accuracy = total_correct_predictions / total_predictions_made * 100
        
        return accuracy

    def Passive_Aggressive(X_train, Y_train, X_test, Y_test):
        ######################  Passive Aggressive ###########################--Code from ASTD
        classifier = PassiveAggressiveClassifier(n_iter=100)
        classifier.fit(X_train, Y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, y_pred)
        if len(cm[0]) == 2:
            total_correct_predictions = cm[0,0]+cm[1,1]
        elif len(cm[0]) == 3:
            total_correct_predictions = cm[0,0]+cm[1,1]+cm[2,2]
        total_predictions_made = np.sum(cm)
        accuracy = total_correct_predictions / total_predictions_made * 100
        
        return accuracy

    def Perceptron(X_train, Y_train, X_test, Y_test):
        ###################### Perceptron  ###################################--Code from ASTD
        classifier = Perceptron(n_iter=100)
        classifier.fit(X_train, Y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, y_pred)
        if len(cm[0]) == 2:
            total_correct_predictions = cm[0,0]+cm[1,1]
        elif len(cm[0]) == 3:
            total_correct_predictions = cm[0,0]+cm[1,1]+cm[2,2]
        total_predictions_made = np.sum(cm)
        accuracy = total_correct_predictions / total_predictions_made * 100
        
        return accuracy

    def bnb(X_train, Y_train, X_test, Y_test):
        ######################  bnb ###########################################--Code from ASTD
        classifier = BernoulliNB(binarize=0.5)
        classifier.fit(X_train, Y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, y_pred)
        if len(cm[0]) == 2:
            total_correct_predictions = cm[0,0]+cm[1,1]
        elif len(cm[0]) == 3:
            total_correct_predictions = cm[0,0]+cm[1,1]+cm[2,2]
        total_predictions_made = np.sum(cm)
        accuracy = total_correct_predictions / total_predictions_made * 100
        
        return accuracy

    def mnb(X_train, Y_train, X_test, Y_test):
        ###################### mnb ############################################--Code from ASTD
        classifier = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        classifier.fit(X_train, Y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, y_pred)
        if len(cm[0]) == 2:
            total_correct_predictions = cm[0,0]+cm[1,1]
        elif len(cm[0]) == 3:
            total_correct_predictions = cm[0,0]+cm[1,1]+cm[2,2]
        total_predictions_made = np.sum(cm)
        accuracy = total_correct_predictions / total_predictions_made * 100
        
        return accuracy
    
    def cnb(X_train, Y_train, X_test, Y_test):
    ##################### CNB ######################
        classifier = ComplementNB()
        #ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
        classifier.fit(X_train, Y_train)
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, y_pred)
        if len(cm[0]) == 2:
            total_correct_predictions = cm[0,0]+cm[1,1]
        elif len(cm[0]) == 3:
            total_correct_predictions = cm[0,0]+cm[1,1]+cm[2,2]
        total_predictions_made = np.sum(cm)
        accuracy = total_correct_predictions / total_predictions_made * 100
        return accuracy

    
    def sgd(X_train, Y_train, X_test, Y_test):
        ################# sgd ###############################################--Code from ASTD
        classifier = SGDClassifier(loss="hinge", penalty="l2")
        classifier.fit(X_train, Y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, y_pred)
        if len(cm[0]) == 2:
            total_correct_predictions = cm[0,0]+cm[1,1]
        elif len(cm[0]) == 3:
            total_correct_predictions = cm[0,0]+cm[1,1]+cm[2,2]
        total_predictions_made = np.sum(cm)
        accuracy = total_correct_predictions / total_predictions_made * 100
        
        return accuracy

    def ANN(X_train, Y_train, X_test, Y_test, input_dim):
        #########################  ANN ##########################---to be checked and take the lesson again 
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
        if len(Y_train[0]) > len(Y_test[0]):
            zero_col = np.zeros(len(Y_test))
            Y_test = np.c_[Y_test, zero_col]
        if len(Y_train[0]) < len(Y_test[0]):
            zero_col = np.zeros(len(Y_train))
            Y_train = np.c_[Y_train, zero_col]
        # Initialising the ANN
        classifier = Sequential()
        classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', 
                             input_dim = input_dim))  
        ## Adding the second hidden layer
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
        # Adding the output layer
        classifier.add(Dense(output_dim = len(Y_train[0]), init = 'uniform', activation = 'softmax'))
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                           metrics = ['accuracy'])
        # Fitting the ANN to the Training set
        classifier.fit(X_train.toarray(), Y_train, batch_size = 128, nb_epoch = 50, verbose=1)
        #######Classifer Evaluation ############old batch size 20
        score = classifier.evaluate(X_test.toarray(), Y_test, batch_size=64, verbose=1)
        return score[1]

    def CNN(X_train, Y_train, X_test, Y_test, input_dim):
        ##################### Convlutional Neural Networks #####################
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
        ## #Building the CNN
        classifier = Sequential()
        if input_dim > 18000:
            top_words = 18000
            max_words = 500
            X_train = sequence.pad_sequences(X_train.toarray(), maxlen=max_words)
            X_test = sequence.pad_sequences(X_test.toarray(), maxlen=max_words)
            classifier.add(Embedding(top_words, 100))
            classifier.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))
            classifier.add(MaxPooling1D(pool_size=2))
            classifier.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))
            classifier.add(MaxPooling1D(pool_size=2))
            classifier.add(Flatten())
            classifier.add(Dense(250, activation='relu'))
            classifier.add(Dense(len(Y_train[0]), activation='softmax'))
            classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=2, batch_size=128, verbose=2)
            # Final evaluation of the model
            scores = classifier.evaluate(X_test, Y_test, verbose=0)
        else:
            classifier.add(Embedding(input_dim, 100))
            classifier.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))
            classifier.add(MaxPooling1D(pool_size=2))
            classifier.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))
            classifier.add(MaxPooling1D(pool_size=2))
            classifier.add(Flatten())
            classifier.add(Dense(250, activation='relu'))
            classifier.add(Dense(len(Y_train[0]), activation='softmax'))
            classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#        print(classifier.summary())
            classifier.fit(X_train.toarray(), Y_train, validation_data=(X_test.toarray(), Y_test), epochs=2, batch_size=128, verbose=2)
            # Final evaluation of the model
            scores = classifier.evaluate(X_test.toarray(), Y_test, verbose=0)
        
        return scores[1]*100

        