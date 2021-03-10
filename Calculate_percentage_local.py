# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 08:21:58 2018
@author: Rana Mahmoud
"""

import os
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu,floatX=float32'

# Parameters matplotlib
# ==================================================
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,10) 
#%matplotlib inline
import seaborn as sns
plt.switch_backend('agg') 

# Parameters General
# ==================================================
import codecs
import csv
import keras
import sklearn
import gensim
import random
import scipy
import pydot
#import commands
import glob
import numpy as np
import pandas as pd
import re

# Parameters theano
# ==================================================
#import theano
#print( 'using : ',(theano.config.device))

# Parameters keras
# ==================================================
#from keras.utils.visualize_util import plot ## to print the model arch
#from keras.utils.vis_utils import plot_model ## to print the model arch
#from keras.preprocessing import sequence
#from keras.preprocessing.text import Tokenizer
##from keras.models import Sequential , Graph
#from keras.models import Sequential
#
#from keras.optimizers import SGD
#from keras.layers.embeddings import Embedding
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.regularizers import l2#, activity_l2
##from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
#from keras.layers import Dense , Dropout , Activation , Merge , Flatten
#from keras.layers.convolutional import Convolution1D, MaxPooling1D
#from keras.layers import Embedding , LSTM
#
#from keras.models import Model
#from keras.layers import Input
#from keras.layers import TimeDistributed
#from keras.layers import Merge, merge
##Merge is for layers, merge is for tensors.
#from keras.utils.vis_utils import plot_model
###LSTM
#from keras.layers.recurrent import LSTM


# Parameters sklearn
# ==================================================
import sklearn.metrics
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC , SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report #get F1 score

# Parameters gensim
# ==================================================
#from gensim.models.word2vec import Word2Vec
#import gensim.models.keyedvectors
#from gensim.models.doc2vec import Doc2Vec , TaggedDocument
#from gensim.corpora.dictionary import Dictionary
#import gensim

import datetime

###Preprocessing libraries
from LoadDataset_General import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from qalsadi import analex
import pyarabic.arabrepr
from tashaphyne.stemming import ArabicLightStemmer
from pyarabic.named import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Model Hyperparameters
# ==================================================
#
print( "Model Hyperparameters :")

embeddings_dim = 300
print ("embeddings_dim = " ,embeddings_dim)

filter_sizes = [3, 5, 7]
print ("filter_sizes = ",filter_sizes)

dropout_prob = [0.5,0.5]
print ("dropout_prob = ",dropout_prob)

# maximum number of words to consider in the representations
max_features = 30000
print ("max_features = " ,max_features)

# percentage of the data used for model training
val_split = 0.75
print ("val_split = ",val_split)


LoadDataset_General = LoadDataset_General()

datasets = list()

datasets = {
        ('ASTD',40), #10000 records
		('BBN',40),
		('SYR',40),
       ('HTL',1110),
        ('MOV',2335),
		('ATT',568),
        ('PROD',234),
       ('RES',539), #10900 records
        ('EG_NU',540),
        ('SemEval',540)
        }

for dataset_name, max_sent_len in datasets:
    # Reading csv data
    # ==================================================
    print ("Reading text data for classification and building representations...")
    reviews = []
#    reviews = [ ( row["text"] , row["polarity"]  ) for row in csv.DictReader(open(file_name, encoding="utf8"), delimiter=',', quoting=csv.QUOTE_NONE) ]
    
    (body_all,rating_all)=LoadDataset_General.Load_Data(dataset_name)
    num_classes = len( set( rating_all ) )

    body = list()
    rating = list()

    columns = {'body': body_all, 'rating': rating_all}
    data = pd.DataFrame(columns, columns = ['body', 'rating'])
    reviews = pd.DataFrame([[body_all, rating_all]])
    pos_vec = list()
    neg_vec = list()
    neut_vec = list()
    pos_vec_train = list()
    neg_vec_train = list()
    neut_vec_train = list()
    pos_vec_test = list()
    neg_vec_test = list()
    neut_vec_test = list()
       
   
    ############### Preprocessing ########
    for i in range(0,len(data)):
        data.iloc[i,0] = re.sub("\"",'',data.iloc[i,0])
        data.iloc[i,0] = LoadDataset_General.Emoticon_detection(data.iloc[i,0])
#        data.iloc[i,0] = re.sub(u'\@',u'', data.iloc[i,0])
        data.iloc[i,0] = LoadDataset_General.clean_raw_review(data.iloc[i,0])
        data.iloc[i,0] = LoadDataset_General.normalizeArabic(data.iloc[i,0])
        data.iloc[i,0] = LoadDataset_General.Elong_remove(data.iloc[i,0])
        data.iloc[i,0] = LoadDataset_General.deNoise(data.iloc[i,0])
#        data.iloc[i,0] = LoadDataset_General.Remove_Stopwords(data.iloc[i,0])
#        data.iloc[i,0] = LoadDataset_General.Named_Entity_Recognition(data.iloc[i,0])
    #    data[i] = LoadDataset_General.Stem_word(data[i])
    #    data.iloc[i,0] = LoadDataset_General.Light_Stem_word(data.iloc[i,0])
    #    data[i] = LoadDataset_General.Get_root_word(data[i])
        if data.iloc[i,1] == 1:
            pos_vec.append(data.iloc[i,1])
        elif data.iloc[i,1] == 2:
            neg_vec.append(data.iloc[i,1])
        elif data.iloc[i,1] == 0:
            neut_vec.append(data.iloc[i,1])

    if dataset_name in ('EG_NU','SemEval'):
        (body,rating)=LoadDataset_General.Load_Data(dataset_name+'_test')
        columns_test = {'body': body, 'rating': rating}
        data_test = pd.DataFrame(columns_test, columns = ['body', 'rating'])
        reviews = pd.DataFrame([[body, rating]])           
            
        
        for i in range(0,len(data_test)):
            data_test.iloc[i,0] = re.sub("\"",'',data_test.iloc[i,0])
            data_test.iloc[i,0] = LoadDataset_General.Emoticon_detection(data_test.iloc[i,0])
#            data_test.iloc[i,0] = re.sub(u'\@',u'', data_test.iloc[i,0])
            data_test.iloc[i,0] = LoadDataset_General.clean_raw_review(data_test.iloc[i,0])
            data_test.iloc[i,0] = LoadDataset_General.normalizeArabic(data_test.iloc[i,0])
            data_test.iloc[i,0] = LoadDataset_General.Elong_remove(data_test.iloc[i,0])
            data_test.iloc[i,0] = LoadDataset_General.deNoise(data_test.iloc[i,0])
#            data_test.iloc[i,0] = LoadDataset_General.Remove_Stopwords(data_test.iloc[i,0])
#            data_test.iloc[i,0] = LoadDataset_General.Named_Entity_Recognition(data_test.iloc[i,0])
            if data_test.iloc[i,1] == 1:
                pos_vec.append(data_test.iloc[i,1])
                pos_vec_test.append(data_test.iloc[i,1])
            elif data_test.iloc[i,1] == 2:
                neg_vec.append(data_test.iloc[i,1])
                neg_vec_test.append(data_test.iloc[i,1])
            elif data_test.iloc[i,1] == 0:
                neut_vec.append(data_test.iloc[i,1])         
                neut_vec_test.append(data_test.iloc[i,1])
        
    if dataset_name not in ('EG_NU','SemEval'):
        train_size = int(len(data) * val_split)
        train_texts = data.iloc[0:train_size,0].tolist()
        test_texts = data.iloc[train_size:-1,0].tolist()
        train_labels = data.iloc[0:train_size,1].tolist()
        test_labels = data.iloc[train_size:-1,1].tolist()
        total_count = len(data)
        train_set_count = len(train_labels)
        test_set_count = len(test_labels)
        pos_vec_train_count = train_labels.count(1)
        neg_vec_train_count = train_labels.count(2)
        neut_vec_train_count = train_labels.count(0)
        pos_vec_test_count = test_labels.count(1)
        neg_vec_test_count = test_labels.count(2)
        neut_vec_test_count = test_labels.count(0)
    else:
        train_texts = data.iloc[:,0].tolist()
        test_texts = data_test.iloc[:,0].tolist()
        train_labels = data.iloc[:,1].tolist()
        test_labels = data_test.iloc[:,1].tolist()
        train_set_count = len(train_labels)
        test_set_count = len(test_labels)
        total_count = len(data) + test_set_count
        pos_vec_train_count = train_labels.count(1)
        neg_vec_train_count = train_labels.count(2)
        neut_vec_train_count = train_labels.count(0)
        pos_vec_test_count = test_labels.count(1)
        neg_vec_test_count = test_labels.count(2)
        neut_vec_test_count = test_labels.count(0)
    
    
    pos_count = len(pos_vec)
    neg_count = len(neg_vec)
    neut_count = len(neut_vec)
    
    with open('E:\\DatasetPercentage.txt', 'a') as the_file:
        the_file.write(dataset_name)
        the_file.write('\n---------------------------------------\n')
        the_file.write('Dataset_Size: ')
        the_file.write(str(total_count))
        the_file.write('\n')
        the_file.write('Positive percentage: ')
        the_file.write(str((pos_count/total_count)*100))
        the_file.write('\n')
        the_file.write('Negative percentage: ')
        the_file.write(str((neg_count/total_count)*100))
        the_file.write('\n')
        the_file.write('Neutral percentage: ')
        the_file.write(str((neut_count/total_count)*100))
        the_file.write('\n')
        the_file.write('Training Set Only')
        the_file.write(str(train_set_count))
        the_file.write('\n')
        the_file.write('Positive percentage: ')
        the_file.write(str((pos_vec_train_count/train_set_count)*100))
        the_file.write('\n')
        the_file.write('Negative percentage: ')
        the_file.write(str((neg_vec_train_count/train_set_count)*100))
        the_file.write('\n')
        the_file.write('Neutral percentage: ')
        the_file.write(str((neut_vec_train_count/train_set_count)*100))
        the_file.write('\n')
        the_file.write('Test Set Only')
        the_file.write(str(test_set_count))
        the_file.write('\n')
        the_file.write('Positive percentage: ')
        the_file.write(str((pos_vec_test_count/test_set_count)*100))
        the_file.write('\n')
        the_file.write('Negative percentage: ')
        the_file.write(str((neg_vec_test_count/test_set_count)*100))
        the_file.write('\n')
        the_file.write('Neutral percentage: ')
        the_file.write(str((neut_vec_test_count/test_set_count)*100))
        the_file.write('\n')