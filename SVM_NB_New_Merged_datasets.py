# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:21:30 2017
@author: PAVILION G4
"""
import datetime
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

datasets = list()

datasets = {
#        ('ASTD',40), #10000 records
		('BBN',40),
		('SYR',40),
       ('HTL',1110),
        ('MOV',2335),
#		('ATT',568),
        ('PROD',234),
       ('RES',539), #10900 records
       ('EG_NU',540),
        ('SemEval',540)
        }

body_all = list()
rating_all = list()
trainset_sizes = list()

val_split = 0.75

an = analex.analex()
tokenizer = an.text_tokenize

token = list()

token = {
        ('count_1', CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 1), max_df=0.01)),
        ('count_2', CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2), max_df=0.01)),
        ('count_3', CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 3), max_df=0.01)),
        ('tfidf1',TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 1), max_df=0.01)),
        ('tfidf2',TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 2), max_df=0.01)),
        ('tfidf3',TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 3), max_df=0.01)),
        }

#Training set
for dataset_name, max_sent_len in datasets:
    print("Merging all training sets")
    reviews = []
    
    if dataset_name not in ('EG_NU','SemEval'):
        (body_tmp1,rating_tmp1)=LoadDataset_General.Load_Data(dataset_name)
        train_size = int(val_split * len(body_tmp1))
        trainset_sizes.append((dataset_name, len(body_tmp1), train_size))
        body_tmp = body_tmp1[0:train_size]
        rating_tmp = rating_tmp1[0:train_size]
    else:
        (body_tmp,rating_tmp)=LoadDataset_General.Load_Data(dataset_name)
        
    body_all.extend(body_tmp)
    rating_all.extend(rating_tmp)

num_classes = len( set( rating_all ) )

body = list()
rating = list()
## TO get 3 classes only
for i in range(0, len(body_all)):
    if rating_all[i] in (0,1,2):
        body.append(body_all[i] )
        rating.append(rating_all[i])

### TO get 2 classes only
#    for i in range(0, len(body_all)):
#        if rating_all[i] != 0:
#            body.append(body_all[i] )
#            rating.append(rating_all[i])

columns = {'body': body, 'rating': rating}
data = pd.DataFrame(columns, columns = ['body', 'rating'])
reviews = pd.DataFrame([[body, rating]])
  
############### Preprocessing ########
for i in range(0,len(data)):
    data.iloc[i,0] = re.sub("\"",'',data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.Emoticon_detection(data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.clean_raw_review(data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.normalizeArabic(data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.Elong_remove(data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.deNoise(data.iloc[i,0])
  

d_train = data.iloc[:,0].tolist()
Y_train = data.iloc[:,1].tolist()

#Test set
for dataset_name, max_sent_len in datasets:
    with open('C:\\Users\\paperspace\\Desktop\\Results\\Results.txt', 'a') as the_file:
        the_file.write(dataset_name)
        the_file.write('\n---------------------------------------\n')

    if dataset_name not in ('EG_NU','SemEval'):
        (body_tmp1,rating_tmp1)=LoadDataset_General.Load_Data(dataset_name)
        train_size = int(val_split * len(body_tmp1))
        body_test = body_tmp1[train_size:]
        rating_test = rating_tmp1[train_size:]
    else:
        (body_test,rating_test)=LoadDataset_General.Load_Data(dataset_name+'_test')
  
    body = list()
    rating = list()
    ## TO get 3 classes only
    for i in range(0, len(body_all)):
        if rating_all[i] in (0,1,2):
            body.append(body_all[i] )
            rating.append(rating_all[i])
    
    ### TO get 2 classes only
    #    for i in range(0, len(body_all)):
    #        if rating_all[i] != 0:
    #            body.append(body_all[i] )
    #            rating.append(rating_all[i])
    
    columns = {'body': body_test, 'rating': rating_test}
    data_testing = pd.DataFrame(columns, columns = ['body', 'rating'])
#    reviews_testing = pd.DataFrame([[body_testing, rating_testing]])

    #Preprocessing
    for i in range(0,len(data_testing)):
        data_testing.iloc[i,0] = re.sub("\"",'',data_testing.iloc[i,0])
        data_testing.iloc[i,0] = LoadDataset_General.Emoticon_detection(data_testing.iloc[i,0])
        data_testing.iloc[i,0] = LoadDataset_General.clean_raw_review(data_testing.iloc[i,0])
        data_testing.iloc[i,0] = LoadDataset_General.normalizeArabic(data_testing.iloc[i,0])
        data_testing.iloc[i,0] = LoadDataset_General.Elong_remove(data_testing.iloc[i,0])
        data_testing.iloc[i,0] = LoadDataset_General.deNoise(data_testing.iloc[i,0])
#        data_test.iloc[i,0] = LoadDataset_General.Remove_Stopwords(data_test.iloc[i,0])
#            data_test.iloc[i,0] = LoadDataset_General.Named_Entity_Recognition(data_test.iloc[i,0])
     
    unbalanced_test_x = data_testing.iloc[:,0].tolist()
    Y_test = data_testing.iloc[:,1].tolist()
    

    ########## Feature Extraction using Tokenization ############
    for tokenizer_name, tf_fg in token:
        with open('C:\\Users\\paperspace\\Desktop\\Results\\Results.txt', 'a') as the_file:
            the_file.write('\nStart_Time\n')
            the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ############# Features using counts and tfidf ######
        X_train = tf_fg.fit_transform(d_train)
        X_test = tf_fg.transform(unbalanced_test_x)

        ############ Classification #############
        ###for lexicon genertion
        classifiers_accuracy = [
        #                        ('KNN', Classifiers.KNN(X_train, Y_train, X_test, Y_test)),
                                ('SVM', Classifiers.SVM(X_train, Y_train, X_test, Y_test)),
        #                        ('Logistic_Regression', Classifiers.Logistic_Regression(X_train, Y_train, X_test, Y_test)),
        #                        ('Passive_Aggressive', Classifiers.Passive_Aggressive(X_train, Y_train, X_test, Y_test)),
        #                        ('Perceptron', Classifiers.Perceptron(X_train, Y_train, X_test, Y_test)),
                                ('bnb', Classifiers.bnb(X_train, Y_train, X_test, Y_test)),
                                ('mnb', Classifiers.mnb(X_train, Y_train, X_test, Y_test)),
        #                        ('sgd', Classifiers.sgd(X_train, Y_train, X_test, Y_test)),
                                ('CNB', Classifiers.cnb(X_train, Y_train, X_test, Y_test))#,
        #                        ('ANN', Classifiers.ANN(X_train, Y_train, X_test, Y_test, input_dim)),
#                                ('CNN', Classifiers.CNN_withEmbed(X_train, Y_train, X_test, Y_test, input_dim))
                                ]
        with open('C:\\Users\\paperspace\\Desktop\\Results\\Results.txt', 'a') as the_file:
            the_file.write(dataset_name)
            the_file.write('\n---------------------------------------\n')
            the_file.write('\nEnd_Time\n')
            the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            the_file.write('\nTokenizer Name\n')
            the_file.write(str(tokenizer_name))
            the_file.write('\n')
        for classifier_name, classifier_accuracy in classifiers_accuracy:
            print(str(classifier_name))
            print(str(classifier_accuracy))
            with open('C:\\Users\\paperspace\\Desktop\\Results\\Results.txt', 'a') as the_file:
                the_file.write('\nClassification Name\n')
                the_file.write(str(classifier_name))
                the_file.write('\n')
                the_file.write('\nClassification Accuracy\n')
                the_file.write(str(classifier_accuracy))
                the_file.write('\n')