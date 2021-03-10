# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:21:30 2017
@author: PAVILION G4
"""
from LoadDataset_General import *
from Lexicon_Generation import *
import codecs
import numpy as np
import pandas as pd
import re
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
import nltk
from nltk.stem.isri import ISRIStemmer
from Classifiers import *
from Feature_Generation import *
import pyarabic.arabrepr
from tashaphyne.stemming import ArabicLightStemmer
from pyarabic.named import *
import sklearn.feature_selection

####### Load dataset ##########
LoadDataset_General = LoadDataset_General()
datasetName = 'BBN'
(body,rating)=LoadDataset_General.Load_Data(datasetName)
      

############ Preprocessing ########
for i in range(0,len(body)):
    body[i] = LoadDataset_General.Emoticon_detection(body[i])
    body[i] = LoadDataset_General.clean_raw_review(body[i])
    body[i] = LoadDataset_General.normalizeArabic(body[i])
    body[i] = LoadDataset_General.Elong_remove(body[i])
    body[i] = LoadDataset_General.deNoise(body[i])
    body[i] = LoadDataset_General.Remove_Stopwords(body[i])
    body[i] = LoadDataset_General.Named_Entity_Recognition(body[i])
#    body[i] = LoadDataset_General.Stem_word(body[i])
    body[i] = LoadDataset_General.Light_Stem_word(body[i])
#    body[i] = LoadDataset_General.Get_root_word(body[i])

#### Load unbalanced dataset
(unbalanced_train_x, unbalanced_train_y, unbalanced_test_x, unbalanced_test_y, unbalanced_valid_x, unbalanced_valid_y) = LoadDataset_General.get_train_test_validation_unbalanced(body, rating, datasetName)
d_train = np.concatenate((unbalanced_train_x, unbalanced_valid_x))
Y_train = np.concatenate((unbalanced_train_y, unbalanced_valid_y)).tolist()
Y_test = unbalanced_test_y

########## Feature Extraction using Tokenization ############
###### Counts ##########
an = analex.analex()
tokenizer = an.text_tokenize
ngram = (1,3)
#tf_fg = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 1))
#tf_fg = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))
tf_fg = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 3))
#tf_fg = CountVectorizer(tokenizer=an.text_tokenize, ngram_range=ngram, min_df=1, max_df=1)

##### tfidf ############
an = analex.analex()
tokenizer = an.text_tokenize
ngram = (1,3)
tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 1))
#tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 2))
#tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 3))
#tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 3), max_features=10000)
#tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=ngram, min_df=1, max_df=1)

############# Features using counts and tfidf ######
X_train = tf_fg.fit_transform(d_train)
X_test = tf_fg.transform(unbalanced_test_x)

########
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import SelectPercentile
#from sklearn.feature_selection import chi2
#from sklearn.feature_selection import mutual_info_classif
#from sklearn.feature_selection import mutual_info_regression
#### Chi2 ####
##X_train = SelectPercentile(chi2, percentile = 80).fit_transform(X_train, Y_train)
##X_test = SelectKBest(chi2, k = X_train.get_shape()[1]).fit_transform(X_test, Y_test)
#### Mutual info
#X_train = SelectPercentile(mutual_info_classif, percentile = 80).fit_transform(X_train, Y_train)
#X_test = SelectKBest(mutual_info_classif, k = X_train.get_shape()[1]).fit_transform(X_test, Y_test)


#### Feature Extraction using Lexicons ############
Lexicon_Generation = Lexicon_Generation()
(posWords,negWords,negationWords,posEmojis,negEmojis) = Lexicon_Generation.getWordLists()
##### Training data
posCount_v = list()
negCount_v = list()
posEmojis_Count_v = list()
negEmojis_Count_v = list()
length_v = list()
for i in range(0,len(d_train)):
    (posCount_t, negCount_t) = Lexicon_Generation.pos_neg_counts(LoadDataset_General,d_train[i], posWords, negWords)
    posCount_v.append(posCount_t)
    negCount_v.append(negCount_t)
    (posEmojis_Count_t, negEmojis_Count_t) = Lexicon_Generation.pos_neg_Emojiss(LoadDataset_General,d_train[i], posEmojis, negEmojis)
    posEmojis_Count_v.append(posEmojis_Count_t)
    negEmojis_Count_v.append(posEmojis_Count_t)
    length_v.append(Lexicon_Generation.Length_Review(LoadDataset_General,d_train[i]))
#### Test Data ############
posCount_t_v = list()
negCount_t_v = list()
posEmojis_Count_t_v = list()
negEmojis_Count_t_v = list()
length_t_v = list()
for i in range(0,len(unbalanced_test_x)):
    (posCount_t, negCount_t) = Lexicon_Generation.pos_neg_counts(LoadDataset_General,unbalanced_test_x[i], posWords, negWords)
    posCount_t_v.append(posCount_t)
    negCount_t_v.append(negCount_t)
    (posEmojis_Count_t, negEmojis_Count_t) = Lexicon_Generation.pos_neg_Emojiss(LoadDataset_General,unbalanced_test_x[i], posEmojis, negEmojis)
    posEmojis_Count_t_v.append(posEmojis_Count_t)
    negEmojis_Count_t_v.append(posEmojis_Count_t)
    length_t_v.append(Lexicon_Generation.Length_Review(LoadDataset_General,unbalanced_test_x[i]))
### Features
X_train= np.c_[posCount_v, negCount_v, posEmojis_Count_v, negEmojis_Count_v, length_v]
X_test= np.c_[posCount_t_v, negCount_t_v, posEmojis_Count_t_v, negEmojis_Count_t_v, length_t_v]
 
############ Classification #############
input_dim = X_train.get_shape()[1]
###for lexicon genertion
#input_dim = len(X_train[0])
classifiers_accuracy = [
#                        ('KNN', Classifiers.KNN(X_train, Y_train, X_test, Y_test)),
#                        ('SVM', Classifiers.SVM(X_train, Y_train, X_test, Y_test)),
#                        ('Logistic_Regression', Classifiers.Logistic_Regression(X_train, Y_train, X_test, Y_test)),
#                        ('Passive_Aggressive', Classifiers.Passive_Aggressive(X_train, Y_train, X_test, Y_test)),
#                        ('Perceptron', Classifiers.Perceptron(X_train, Y_train, X_test, Y_test)),
#                        ('bnb', Classifiers.bnb(X_train, Y_train, X_test, Y_test)),
#                        ('mnb', Classifiers.mnb(X_train, Y_train, X_test, Y_test)),
#                        ('sgd', Classifiers.sgd(X_train, Y_train, X_test, Y_test)),
#                        ('CNB', Classifiers.cnb(X_train, Y_train, X_test, Y_test))
#                        ,
#                        ('ANN', Classifiers.ANN(X_train, Y_train, X_test, Y_test, input_dim))
#                        ,
                        ('CNN', Classifiers.CNN_withEmbed(X_train, Y_train, X_test, Y_test, input_dim))
                        ]
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
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
input_dim = X_train.get_shape()[1]
classifier = Sequential()
classifier.add(Embedding(input_dim, 100))
print(classifier.output_shape)
classifier.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))
print(classifier.output_shape)
classifier.add(MaxPooling1D(pool_size=2))
print(classifier.output_shape)
classifier.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))
print(classifier.output_shape)
classifier.add(MaxPooling1D(pool_size=2))
print(classifier.output_shape)
classifier.add(Flatten())
print(classifier.output_shape)
classifier.add(Dense(250, activation='relu'))
print(classifier.output_shape)
classifier.add(Dense(len(Y_train[0]), activation='softmax'))
print(classifier.output_shape)
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#        print(classifier.summary())
classifier.fit(X_train.toarray(), Y_train, validation_data=(X_test.toarray(), Y_test), epochs=10, batch_size=64, verbose=2)
# Final evaluation of the model
scores = classifier.evaluate(X_test.toarray(), Y_test, verbose=0)


classifier = Sequential()
classifier.add(Embedding(100, 32, input_length=input_dim))
print(classifier.output_shape)
classifier.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
print(classifier.output_shape)
classifier.add(MaxPooling1D(pool_size=2))
print(classifier.output_shape)
classifier.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
print(classifier.output_shape)
classifier.add(MaxPooling1D(pool_size=2))
print(classifier.output_shape)
#classifier.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
#print(classifier.output_shape)
#classifier.add(MaxPooling1D(pool_size=2))
#print(classifier.output_shape)
classifier.add(Flatten())
print(classifier.output_shape)
classifier.add(Dense(250, activation='relu'))
print(classifier.output_shape)
classifier.add(Dropout(0.5))
print(classifier.output_shape)
classifier.add(Dense(len(Y_train[0]), activation='softmax'))
print(classifier.output_shape)
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#        print(classifier.summary())
classifier.fit(X_train.toarray(), Y_train, validation_data=(X_test.toarray(), Y_test), epochs=10, batch_size=64, verbose=2)
# Final evaluation of the model
scores = classifier.evaluate(X_test.toarray(), Y_test, verbose=0)
print(scores[1]*100)

scores = classifier.evaluate(X_train.toarray(), Y_train, verbose=0)
print(scores[1]*100)

num_classes = len( set( Y_train + Y_test) )
model = Sequential()
lay1 = model.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', 
                     input_dim = input_dim))  
## Adding the second hidden layer
lay2 = model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
if num_classes == 2:
    lay3 = model.add(Dense(output_dim = 1, init = 'uniform', activation = 'softmax'))
else:
    lay3 = model.add(Dense(output_dim = num_classes, init = 'uniform', activation = 'softmax'))

outputs    = [layer.output for layer in model.layers] 
inputs    = [layer.input for layer in model.layers]

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
# Fitting the ANN to the Training set
model.fit(X_train.toarray(), Y_train, batch_size = 128, nb_epoch = 50, verbose=1)
#######Classifer Evaluation ######
score = model.evaluate(X_test.toarray(), Y_test, batch_size=64, verbose=1)
accuracy = score[1]

########## Lexicon Classification
Lexicon_Generation = Lexicon_Generation()
(posWords,negWords,negationWords,posEmojis,negEmojis) = Lexicon_Generation.getWordLists()
SentCount_v = list()
for i in range(0,len(body)):
    SentCount= Lexicon_Generation.Lexicon_Sentiment(body[i], posWords,negWords)
    if SentCount > 0:
        SentCount_v.append(1)
    elif SentCount < 0:
        SentCount_v.append(-1)
    else:
        SentCount_v.append(0)
### Evaluation
true_sent=0            
for i in range(0, len(SentCount_v)):
    if SentCount_v[i] == rating[i]:
         true_sent +=1                
accuracy =true_sent/len(SentCount_v)
