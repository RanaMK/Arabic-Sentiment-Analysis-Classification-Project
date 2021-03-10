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
       ('ASTD',40), #10000 records
#		('BBN',40),
#		('SYR',40),
       ('HTL',1110),
       ('MOV',2335),
		('ATT',568),
       ('PROD',234),
       ('RES',539)#, #10900 records
#       ('EG_NU',540),
#       ('SemEval',540)
        }

Original_datasets = {
        'ASTD', #10000 records
#		'BBN',
#		'SYR',
       'HTL',
       'MOV',
		'ATT',
       'PROD',
       'RES'#, #10900 records
#       'EG_NU',
#       'SemEval'
        }

body_all = list()
rating_all = list()
body_tmp = list()
rating_tmp = list()
body_test = list()
rating_test = list()

val_split = 0.75

an = analex.analex()
tokenizer = an.text_tokenize

token = list()

token = {
        ('count_1', CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 1))),
        ('count_2', CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))),
        ('count_3', CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 3))),
        ('tfidf1',TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 1))),
        ('tfidf2',TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 2))),
        ('tfidf3',TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 3))),
        }

  
for dataset_name, max_sent_len in datasets:
    # Reading csv data
    # ==================================================
    print ("Reading text data for classification and building representations...")
    reviews = []
    for original_dataset in Original_datasets:
        if original_dataset != dataset_name:
            (body_tmp,rating_tmp)=LoadDataset_General.Load_Data(original_dataset)
        else:
            (body_tmp1,rating_tmp1)=LoadDataset_General.Load_Data(original_dataset)
            train_size = int(val_split * len(body_tmp1))
            body_tmp = body_tmp1[0:train_size]
            rating_tmp = rating_tmp1[0:train_size]
            body_test = body_tmp1[train_size:]
            rating_test = rating_tmp1[train_size:]
            body_testing = list()
            rating_testing = list()
            for i in range(0, len(body_test)):
                if rating_test[i] != 0:
                    body_testing.append(body_test[i] )
                    rating_testing.append(rating_test[i])
            columns = {'body': body_testing, 'rating': rating_testing}
            data_test = pd.DataFrame(columns, columns = ['body', 'rating'])
        body_all.extend(body_tmp)
        rating_all.extend(rating_tmp)

#    num_classes = len( set( rating_all ) )

    body = list()
    rating = list()
    for i in range(0, len(body_all)):
        if rating_all[i] != 0:
            body.append(body_all[i] )
            rating.append(rating_all[i])
    
    columns = {'body': body_all, 'rating': rating_all}
    data = pd.DataFrame(columns, columns = ['body', 'rating'])
    reviews = pd.DataFrame([[body_all, rating_all]])
        
    
    ############### Preprocessing ########
    for i in range(0,len(data)):
        data.iloc[i,0] = re.sub("\"",'',data.iloc[i,0])
        data.iloc[i,0] = LoadDataset_General.Emoticon_detection(data.iloc[i,0])
        data.iloc[i,0] = LoadDataset_General.clean_raw_review(data.iloc[i,0])
        data.iloc[i,0] = LoadDataset_General.normalizeArabic(data.iloc[i,0])
        data.iloc[i,0] = LoadDataset_General.Elong_remove(data.iloc[i,0])
        data.iloc[i,0] = LoadDataset_General.deNoise(data.iloc[i,0])
#        data.iloc[i,0] = LoadDataset_General.Remove_Stopwords(data.iloc[i,0])
#        data.iloc[i,0] = LoadDataset_General.Named_Entity_Recognition(data.iloc[i,0])
    #    data[i] = LoadDataset_General.Stem_word(data[i])
    #    data.iloc[i,0] = LoadDataset_General.Light_Stem_word(data.iloc[i,0])
    #    data[i] = LoadDataset_General.Get_root_word(data[i])

    #### Load unbalanced dataset
    train_size = int(len(data) * val_split)
    d_train = data.iloc[0:train_size,0].tolist()
    unbalanced_test_x = data.iloc[train_size:-1,0].tolist()
    Y_train = data.iloc[0:train_size,1].tolist()
    Y_test = data.iloc[train_size:-1,1].tolist()
    num_classes = len( set( Y_train + Y_test ) )
    

    ########## Feature Extraction using Tokenization ############
    for tokenizer_name, tf_fg in token:
        with open('C:\\Users\\paperspace\\Desktop\\Results\\Results.txt', 'a') as the_file:
            the_file.write('\nStart_Time\n')
            the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ############# Features using counts and tfidf ######
        X_train = tf_fg.fit_transform(d_train)
        X_test = tf_fg.transform(unbalanced_test_x)

        ############ Classification #############
        input_dim = X_train.get_shape()[1]
        ###for lexicon genertion
        #input_dim = len(X_train[0])
        classifiers_accuracy = [
        #                        ('KNN', Classifiers.KNN(X_train, Y_train, X_test, Y_test)),
#                                ('SVM', 
                                 Classifiers.SVM(X_train, Y_train, X_test, Y_test)
                                 #),
        #                        ('Logistic_Regression', Classifiers.Logistic_Regression(X_train, Y_train, X_test, Y_test)),
        #                        ('Passive_Aggressive', Classifiers.Passive_Aggressive(X_train, Y_train, X_test, Y_test)),
        #                        ('Perceptron', Classifiers.Perceptron(X_train, Y_train, X_test, Y_test)),
        #                        ('bnb', Classifiers.bnb(X_train, Y_train, X_test, Y_test)),
        #                        ('mnb', Classifiers.mnb(X_train, Y_train, X_test, Y_test)),
        #                        ('sgd', Classifiers.sgd(X_train, Y_train, X_test, Y_test)),
#                                ('CNB', Classifiers.cnb(X_train, Y_train, X_test, Y_test))#,
        #                        ('ANN', Classifiers.ANN(X_train, Y_train, X_test, Y_test, input_dim)),
#                                ('CNN', Classifiers.CNN_withEmbed(X_train, Y_train, X_test, Y_test, input_dim))
                                ]
        with open('C:\\Users\\paperspace\\Desktop\\Results\\Results.txt', 'a') as the_file:
#        with open('E:\\Results.txt', 'a') as the_file:
            the_file.write(dataset_name)
            the_file.write('\n---------------------------------------\n')
            the_file.write('\nEnd_Time\n')
            the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            the_file.write('Dataset_Size: ')
            the_file.write(str(len(body)))
            the_file.write('And after removing duplicates: ')
            the_file.write(str(len(data)))
            the_file.write('\n')
            the_file.write('Number of Classes: ')
            the_file.write(str(num_classes))
            the_file.write('\n')
            the_file.write('\nSVM\n')
            the_file.write('\nTokenizer Name\n')
            the_file.write(str(tokenizer_name))
            the_file.write('\n')
            the_file.write('\nClassification Accuracy\n')
            the_file.write(str(classifiers_accuracy))
            the_file.write('\n')