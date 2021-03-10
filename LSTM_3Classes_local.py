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
from keras.utils.vis_utils import plot_model ## to print the model arch
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
#from keras.models import Sequential , Graph
from keras.models import Sequential

from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2#, activity_l2
#from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
from keras.layers import Dense , Dropout , Activation , Merge , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM

from keras.models import Model
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Merge, merge
#Merge is for layers, merge is for tensors.
from keras.utils.vis_utils import plot_model
##LSTM
from keras.layers.recurrent import LSTM


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
from gensim.models.word2vec import Word2Vec
import gensim.models.keyedvectors
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from gensim.corpora.dictionary import Dictionary
import gensim

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


# number of classes
num_classes = 2
print ("num_classes = " ,num_classes)

# Training parameters
# ==================================================
num_epochs = 30
print ("num_epochs = ",num_epochs)


    
# Reading pre-trained word embeddings
# ==================================================
# load the embeddings
print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
#embeddings = Word2Vec.load_word2vec_format("C:\\Users\\paperspace\\Desktop\\(CBOW58)-ASA-3B-CBOW-window5-3iter-d300-vecotrs.bin", binary=True,encoding='utf8', unicode_errors='ignore')
#embeddings = gensim.models.KeyedVectors.load_word2vec_format("C:\\Users\\paperspace\\Desktop\\(CBOW58)-ASA-3B-CBOW-window5-3iter-d300-vecotrs.bin", binary=True,encoding='utf8', unicode_errors='ignore')
embeddings = gensim.models.KeyedVectors.load_word2vec_format("E:\\data\\cbow\\(CBOW58)-ASA-3B-CBOW-window5-3iter-d300-vecotrs.bin", binary=True,encoding='utf8', unicode_errors='ignore')
#embeddings = gensim.models.Word2Vec.load('C:\\Users\\paperspace\\Desktop\\Twt-CBOW\\Twt-CBOW')
#embeddings = gensim.models.Word2Vec.load('E:\\data\\aravec\\Twt-CBOW')


LoadDataset_General = LoadDataset_General()

datasets = list()

datasets = {
#        ('ASTD',40), #10000 records
#		('BBN',40)#,
#		('SYR',40)#,
#       ('HTL',1110),
#        ('MOV',2335)#,
#		('ATT',568)#,
#        ('PROD',234),
       ('RES',539)#, #10900 records
#        ('EG_NU',540)#,
#        ('SemEval',540)
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
#    for i in range(0, len(body_all)):
#        if rating_all[i] != 0:
#            body.append(body_all[i] )
#            rating.append(rating_all[i])
#    
    columns = {'body': body_all, 'rating': rating_all}
    data = pd.DataFrame(columns, columns = ['body', 'rating'])
    reviews = pd.DataFrame([[body_all, rating_all]])
       
   
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
         

#    random.shuffle( data )
    if dataset_name not in ('EG_NU','SemEval'):
        train_size = int(len(data) * val_split)
        train_texts = data.iloc[0:train_size,0].tolist()
        test_texts = data.iloc[train_size:-1,0].tolist()
        train_labels = data.iloc[0:train_size,1].tolist()
        test_labels = data.iloc[train_size:-1,1].tolist()
        num_classes = len( set( train_labels + test_labels ) )
        tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ")
        tokenizer.fit_on_texts(train_texts)
        train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=max_sent_len )
        test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=max_sent_len )
        train_matrix = tokenizer.texts_to_matrix( train_texts )
        test_matrix = tokenizer.texts_to_matrix( test_texts )
        embedding_weights = np.zeros( ( max_features , embeddings_dim ) )
        for word,index in tokenizer.word_index.items():
            if index < max_features:
                try: embedding_weights[index,:] = embeddings[word]
                except: embedding_weights[index,:] = np.random.uniform(-0.25,0.25,embeddings_dim)
        le = preprocessing.LabelEncoder( )
        le.fit( train_labels + test_labels )
        train_labels = le.transform( train_labels )
        test_labels = le.transform( test_labels )
    else:
        train_texts = data.iloc[:,0].tolist()
        test_texts = data_test.iloc[:,0].tolist()
        train_labels = data.iloc[:,1].tolist()
        test_labels = data_test.iloc[:,1].tolist()
        num_classes = len( set( train_labels + test_labels ) )
        tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ")
        tokenizer.fit_on_texts(train_texts)
        train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=max_sent_len )
        test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=max_sent_len )
        train_matrix = tokenizer.texts_to_matrix( train_texts )
        test_matrix = tokenizer.texts_to_matrix( test_texts )
        embedding_weights = np.zeros( ( max_features , embeddings_dim ) )
        for word,index in tokenizer.word_index.items():
            if index < max_features:
                try: embedding_weights[index,:] = embeddings[word]
                except: embedding_weights[index,:] = np.random.uniform(-0.25,0.25,embeddings_dim)
        le = preprocessing.LabelEncoder( )
        le.fit( train_labels + test_labels )
        train_labels = le.transform( train_labels )
        test_labels = le.transform( test_labels )
        
    print ("Classes : " + repr( le.classes_ ))

    if num_classes > 2:    
        labelencoder_train_labels = LabelEncoder()
        train_labels = labelencoder_train_labels.fit_transform(train_labels)
        #train_labels = keras.utils.to_categorical(train_labels, 4)
        onehotencoder = OneHotEncoder(categorical_features = [0])
        train_labels = train_labels.reshape((-1, 1))
        train_labels = onehotencoder.fit_transform(train_labels).toarray()
        labelencoder_test_labels = LabelEncoder()
        test_labels = labelencoder_test_labels.fit_transform(test_labels)
        onehotencoder = OneHotEncoder(categorical_features = [0])
        test_labels = test_labels.reshape((-1, 1))
        test_labels = onehotencoder.fit_transform(test_labels).toarray() 

    ##LSTM without dropout
    # fix random seed for reproducibility
    np.random.seed(7)

    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] ))
    model.add(LSTM(100))
    if num_classes ==2:
        model.add(Dense(1, activation = 'sigmoid'))
    else:
        model.add(Dense(num_classes, activation = 'sigmoid'))
    if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy']) 
    else: model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    #   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

  
#    plot_model(model, to_file='C:\\Users\\paperspace\\Desktop\\Plots\\model_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(model, to_file='E:\\Results_3Classes\\model_plot_LSTM_1.png', show_shapes=True, show_layer_names=True)
  
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write(dataset_name)
        the_file.write('\n---------------------------------------\n')
        the_file.write('Dataset_Size: ')
        the_file.write(str(len(reviews)))
        the_file.write('And after removing duplicates: ')
        the_file.write(str(len(data)))
        the_file.write('\n')
        the_file.write('Number of Classes: ')
        the_file.write(str(num_classes))
#        the_file.write('Classes are: ')
#        the_file.write(repr( le.classes_ ))
        the_file.write('\n')
        the_file.write('\nStart_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')

    ##fit with writing to a file
#    pd.DataFrame(model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2).history).to_csv("C:\\Users\\paperspace\\Desktop\\Results\\history_"+dataset_name+".csv")
    
    model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2)      

#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write('\nEnd_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')
    
    # Evaluate model keras
    # ==================================================
    print("Evaluate...")
    score, acc  = model.evaluate(test_sequences, test_labels,batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write('\nLSTM without dropout\n')
        the_file.write('\nTest score: ')
        the_file.write(str(score))
        the_file.write('\nTest accuracy: ')
        the_file.write(str(acc))
        the_file.write('\n')
    
    ##LSTM with dropout
    # fix random seed for reproducibility
    np.random.seed(7)

    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] ))
    model.add(Dropout(dropout_prob[1]))
    model.add(LSTM(100))
    model.add(Dropout(dropout_prob[1]))
    if num_classes ==2:
        model.add(Dense(1, activation = 'sigmoid'))
    else:
        model.add(Dense(num_classes, activation = 'sigmoid'))
    if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy']) 
    else: model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    #   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

  
#    plot_model(model, to_file='C:\\Users\\paperspace\\Desktop\\Plots\\model_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(model, to_file='E:\\Results_3Classes\\model_plot_LSTM_2.png', show_shapes=True, show_layer_names=True)
  
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write(dataset_name)
        the_file.write('\n---------------------------------------\n')
        the_file.write('Dataset_Size: ')
        the_file.write(str(len(reviews)))
        the_file.write('And after removing duplicates: ')
        the_file.write(str(len(data)))
        the_file.write('\n')
        the_file.write('Number of Classes: ')
        the_file.write(str(num_classes))
#        the_file.write('Classes are: ')
#        the_file.write(repr( le.classes_ ))
        the_file.write('\n')
        the_file.write('\nStart_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')

    ##fit with writing to a file
#    pd.DataFrame(model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2).history).to_csv("C:\\Users\\paperspace\\Desktop\\Results\\history_"+dataset_name+".csv")
    pd.DataFrame(model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2).history).to_csv("E:\\Results_3Classes\\Weights\\history_"+dataset_name+".csv")
    
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write('\nEnd_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')
    
    # Evaluate model keras
    # ==================================================
    print("Evaluate...")
    score, acc  = model.evaluate(test_sequences, test_labels,batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write('\nLSTM with dropout\n')
        the_file.write('\nTest score: ')
        the_file.write(str(score))
        the_file.write('\nTest accuracy: ')
        the_file.write(str(acc))
        the_file.write('\n')
   
        ##LSTM with dropout and recurrent dropout
    # fix random seed for reproducibility
    np.random.seed(7)

    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] ))
    model.add(LSTM(100, dropout=dropout_prob[1], recurrent_dropout=dropout_prob[1]))
    if num_classes ==2:
        model.add(Dense(1, activation = 'sigmoid'))
    else:
        model.add(Dense(num_classes, activation = 'sigmoid'))
    if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy']) 
    else: model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    #   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

  
#    plot_model(model, to_file='C:\\Users\\paperspace\\Desktop\\Plots\\model_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(model, to_file='E:\\Results_3Classes\\model_plot_LSTM_3.png', show_shapes=True, show_layer_names=True)
  
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write(dataset_name)
        the_file.write('\n---------------------------------------\n')
        the_file.write('Dataset_Size: ')
        the_file.write(str(len(reviews)))
        the_file.write('And after removing duplicates: ')
        the_file.write(str(len(data)))
        the_file.write('\n')
        the_file.write('Number of Classes: ')
        the_file.write(str(num_classes))
#        the_file.write('Classes are: ')
#        the_file.write(repr( le.classes_ ))
        the_file.write('\n')
        the_file.write('\nStart_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')

    ##fit with writing to a file
#    pd.DataFrame(model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2).history).to_csv("C:\\Users\\paperspace\\Desktop\\Results\\history_"+dataset_name+".csv")
    pd.DataFrame(model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2).history).to_csv("E:\\Results_3Classes\\Weights\\history_"+dataset_name+".csv")
    
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write('\nEnd_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')
    
    # Evaluate model keras
    # ==================================================
    print("Evaluate...")
    score, acc  = model.evaluate(test_sequences, test_labels,batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write('\nLSTM with dropout an recurrent dropout\n')
        the_file.write('\nTest score: ')
        the_file.write(str(score))
        the_file.write('\nTest accuracy: ')
        the_file.write(str(acc))
        the_file.write('\n')

    ##LSTM and cnn
    # fix random seed for reproducibility
    np.random.seed(7)

    embedding_vecor_length = 32
    model = Sequential()
    #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] ))
    model.add(Convolution1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    if num_classes ==2:
        model.add(Dense(1, activation = 'sigmoid'))
    else:
        model.add(Dense(num_classes, activation = 'sigmoid'))
    if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy']) 
    else: model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    #   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

  
#    plot_model(model, to_file='C:\\Users\\paperspace\\Desktop\\Plots\\model_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(model, to_file='E:\\Results_3Classes\\model_plot_LSTM_4.png', show_shapes=True, show_layer_names=True)
  
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write(dataset_name)
        the_file.write('\n---------------------------------------\n')
        the_file.write('Dataset_Size: ')
        the_file.write(str(len(reviews)))
        the_file.write('And after removing duplicates: ')
        the_file.write(str(len(data)))
        the_file.write('\n')
        the_file.write('Number of Classes: ')
        the_file.write(str(num_classes))
#        the_file.write('Classes are: ')
#        the_file.write(repr( le.classes_ ))
        the_file.write('\n')
        the_file.write('\nStart_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')

    ##fit with writing to a file
#    pd.DataFrame(model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2).history).to_csv("C:\\Users\\paperspace\\Desktop\\Results\\history_"+dataset_name+".csv")
    pd.DataFrame(model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2).history).to_csv("E:\\Results_3Classes\\Weights\\history_"+dataset_name+".csv")
    
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write('\nEnd_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')
    
    # Evaluate model keras
    # ==================================================
    print("Evaluate...")
    score, acc  = model.evaluate(test_sequences, test_labels,batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    with open('E:\\Results_3Classes\\Results.txt', 'a') as the_file:
        the_file.write('\nCNN then LSTM\n')
        the_file.write('\nTest score: ')
        the_file.write(str(score))
        the_file.write('\nTest accuracy: ')
        the_file.write(str(acc))
        the_file.write('\n')
