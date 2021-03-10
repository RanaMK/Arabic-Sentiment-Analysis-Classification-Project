# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:21:30 2017
@author: PAVILION G4
"""
import codecs
import numpy as np
import pandas as pd
import re
from Load_Dataset_HTLs import *
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
#from sklearn.feature_selection import 


####### Load dataset ##########
LoadDataset_HTLs=LoadDataset_HTLs()
(body,rating)=LoadDataset_HTLs.Load_data()

############ Preprocessing ########
for i in range(0,len(body)):
    body[i] = LoadDataset_HTLs.clean_raw_review(body[i])
    body[i] = LoadDataset_HTLs.normalizeArabic(body[i])
    body[i] = LoadDataset_HTLs.Elong_remove(body[i])
    body[i] = LoadDataset_HTLs.deNoise(body[i])
    body[i] = LoadDataset_HTLs.Named_Entity_Recognition(body[i])
#    body[i] = LoadDataset_HTLs.Stem_word(body[i])
    body[i] = LoadDataset_HTLs.Light_Stem_word(body[i])
#    body[i] = LoadDataset_HTLs.Get_root_word(body[i])


#### Load unbalanced dataset
(unbalanced_train_x, unbalanced_train_y, unbalanced_test_x, unbalanced_test_y, unbalanced_valid_x, unbalanced_valid_y) = LoadDataset_HTLs.get_train_test_validation_unbalanced(body, rating)
d_train = np.concatenate((unbalanced_train_x, unbalanced_valid_x))
Y_train = np.concatenate((unbalanced_train_y, unbalanced_valid_y))
Y_test = unbalanced_test_y

########## Feature Selection ############
###### Counts ##########
# tokenizer
an = analex.analex()
tokenizer = an.text_tokenize
#tf_fg = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 1))
tf_fg = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))
#tf_fg = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 3))

##### tfidf ############
# tokenizer
an = analex.analex()
tokenizer = an.text_tokenize
#tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 1))
#tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 2))
tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 3))

Features = tf_fg.fit_transform(Lex_Words)
X_train = tf_fg.transform(d_train)
X_test = tf_fg.transform(unbalanced_test_x)
tf_fg.get_feature_names()
############# Features using counts and tfidf ######
X_train = tf_fg.fit_transform(d_train)
X_test = tf_fg.transform(unbalanced_test_x)


#### Lexicons ############
Feature_Generation = Feature_Generation()
body_final = list()
posCount_v = list()
negCount_v = list()
posEmojis_Count_v = list()
negEmojis_Count_v = list()
for i in range(0,len(d_train)):
    (lexicon, posCount, negCount, posEmojis_Count, negEmojis_Count, body_txt) = Feature_Generation.Lexicon_generation(LoadDataset_HTLs,d_train[i])
    body_final.append(body_txt)
    posCount_v.append(posCount)
    negCount_v.append(negCount)
    posEmojis_Count_v.append(posEmojis_Count)
    negEmojis_Count_v.append(posEmojis_Count)

#X_train = np.array()
X_train= np.c_[posCount_v, negCount_v, posEmojis_Count_v, negEmojis_Count_v]
 
############# Features using lexicons ######
an = analex.analex()
tokenizer = an.text_tokenize

lexicon_vectorizer1 = TfidfVectorizer(tokenizer=tokenizer,min_df=1,vocabulary=lexicon,ngram_range=(1, 3),binary=True)
X_lexicon_train1 = lexicon_vectorizer1.fit_transform(body_final)
X_lexicon_test1 = lexicon_vectorizer1.transform(unbalanced_test_x)
#X_train1=hstack((X_lexicon_train1, X_lexicon_train2))
X_train=X_lexicon_train1
#X_test1=hstack((X_lexicon_test1, X_lexicon_test2))
X_test=X_lexicon_test1
X_train=X_train.tocsr()
X_test=X_test.tocsr() 

X_train= np.c_[posCount_v, negCount_v]
#X_test = list()
X_test= np.c_[posCount_t_v, negCount_t_v]

X_train= np.c_[posCount_v, negCount_v, posEmojis_Count_v, negEmojis_Count_v]
X_test= np.c_[posCount_t_v, negCount_t_v, posEmojis_Count_t_v, negEmojis_Count_t_v]

X_test=list()
X_train = np.asarray(SentCount_v).reshape(-1,1)
X_test = np.asarray(SentCount_t_v).reshape(-1,1)
Y_test = np.asarray(Y_test).reshape(-1,1)
Y_train = np.asarray(Y_train).reshape(-1,1)

############ Classification #############
classifiers_accuracy = [('KNN', Classifiers.KNN(X_train, Y_train, X_test, Y_test)),
                        ('SVM', Classifiers.SVM(X_train, Y_train, X_test, Y_test)),
                        ('Logistic_Regression', Classifiers.Logistic_Regression(X_train, Y_train, X_test, Y_test)),
                        ('Perceptron', Classifiers.Perceptron(X_train, Y_train, X_test, Y_test)),
                        ('bnb', Classifiers.bnb(X_train, Y_train, X_test, Y_test)),
                        ('mnb', Classifiers.mnb(X_train, Y_train, X_test, Y_test)),
                        ('CNB', Classifiers.cnb(X_train, Y_train, X_test, Y_test)),
                        ('sgd', Classifiers.sgd(X_train, Y_train, X_test, Y_test))
#                        ,
#                        ('SVM', Classifiers.ANN(X_train, Y_train, X_test, Y_test)),
#                        ('SVM', Classifiers.CNN(X_train, Y_train, X_test, Y_test))
                        ]


classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
total_correct_predictions = cm[0,0]+cm[1,1]+cm[2,2]
total_predictions_made = np.sum(cm)
accuracy = total_correct_predictions / total_predictions_made * 100

############ test
lexicon_path="E:\\data\\Lexicons\\"
lexicon_path_lex="E:\\data\\Lexicons\\dr_samha_lex\\"
posWords=pd.read_csv(lexicon_path_lex+'Pos.txt')
negWords=pd.read_csv(lexicon_path_lex+'Neg.txt')
#        negFileAdd=pd.read_csv(lexicon_path+'neg_words_all.txt')
stopWords=pd.read_csv(lexicon_path+'stop_words.txt')
negationWords=pd.read_csv(lexicon_path+'negation_words.txt')
posEmojis=pd.read_csv(lexicon_path+'pos_emojis.txt')
negEmojis=pd.read_csv(lexicon_path+'neg_emojis.txt')
posSent_col = np.ones(len(posWords), dtype=int).reshape(-1,1)
negSent_col = np.full(len(negWords), -1, dtype=int).reshape(-1,1)
posWords = np.c_[posWords, posSent_col]
negWords = np.c_[negWords, negSent_col]
Lexicon = pd.DataFrame(np.vstack((posWords,negWords)))

for i in range(0,len(Lexicon)):
    Lexicon[0][i] = LoadDataset_HTLs.normalizeArabic(Lexicon[0][i])
    Lexicon[0][i] = LoadDataset_HTLs.Elong_remove(Lexicon[0][i])
    Lexicon[0][i] = LoadDataset_HTLs.Light_Stem_word(Lexicon[0][i])  
##Remove deuplication
Lexicon = pd.DataFrame(Lexicon.drop_duplicates().values)

Lex_Words = Lexicon.iloc[:,0].values.tolist()
Lex_Words_S = Lexicon.iloc[:,1].values.tolist()
from nltk import bigrams
from nltk import trigrams
SentCount = 0
SentCount_v = list()
############ Pos and Neg Words ######
for i in range(0,len(body)):
    word = body[i].split(u" ")
    word_final = list()
    SentCount = 0
    for w in word:
        if w in Lex_Words:
            SentCount+=Lex_Words_S[Lex_Words.index(w)]

    bigram_body=list(bigrams(body[i].split()))
    word = [x[0]+" "+x[1] for x in bigram_body]    
    for w in word:
        if w in Lex_Words:
            SentCount+=Lex_Words_S[Lex_Words.index(w)]
            
    trigram_body=list(trigrams(body[i].split()))
    word = [x[0]+" "+x[1] for x in trigram_body]
    for w in word:
        if w in Lex_Words:
            SentCount+=Lex_Words_S[Lex_Words.index(w)]
            
    if SentCount > 0:
        SentCount_v.append(1)
    elif SentCount < 0:
        SentCount_v.append(-1)
    else:
        SentCount_v.append(0)

true_sent=0            
for i in range(0, len(SentCount_v)):
    if SentCount_v[i] == rating[i]:
        true_sent +=1
        
accuracy =true_sent/len(SentCount_v)