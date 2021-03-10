# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 08:21:58 2018

@author: PAVILION G4
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

import datetime


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
#
num_epochs = 30
print ("num_epochs = ",num_epochs)


    
# Reading pre-trained word embeddings
# ==================================================
#   
print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
#embeddings = Word2Vec.load_word2vec_format("E:\data\cbow\(CBOW58)-ASA-3B-CBOW-window5-3iter-d300-vecotrs.bin", binary=True,encoding='utf8', unicode_errors='ignore')
embeddings = gensim.models.KeyedVectors.load_word2vec_format("E:\data\cbow\(CBOW58)-ASA-3B-CBOW-window5-3iter-d300-vecotrs.bin", binary=True,encoding='utf8', unicode_errors='ignore')
#Word vector storage and similarity look-ups. Common code independent of the way the vectors are trained(Word2Vec, FastText, WordRank, VarEmbed etc)
#The word vectors are considered read-only in this class.
#The vectors can be instantiated from an existing file on disk in the original Google’s word2vec C format as a KeyedVectors instance
#.bin --> # C binary format
#load_word2vec_format(fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict', limit=None, datatype=<type 'numpy.float32'>)
#Load the input-hidden weight matrix from the original C word2vec-tool format.
#Note that the information stored in the file is incomplete (the binary tree is missing), so while you can query for word similarity etc., you cannot continue training with a model loaded this way.
#binary is a boolean indicating whether the data is in binary word2vec format. 
#If you trained the C model using non-utf8 encoding for words, specify that encoding in encoding.
#unicode_errors, default ‘strict’, is a string suitable to be passed as the errors argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source file may include word tokens truncated in the middle of a multibyte unicode character (as is common from the original word2vec.c tool), ‘ignore’ or ‘replace’ may help.

#The .bin file is produced using word2vec tool in link below
#https://code.google.com/archive/p/word2vec/
#The word2vec tool takes a text corpus as input and produces the word vectors as output. It first constructs a vocabulary from the training text data and then learns vector representation of words. The resulting word vector file can be used as features in many natural language processing and machine learning applications.
#There are two main learning algorithms in word2vec : continuous bag-of-words and continuous skip-gram. The switch -cbow allows the user to pick one of these learning algorithms. Both algorithms learn the representation of a word that is useful for prediction of other words in the sentence. 
#Using the word2vec tool, it is possible to train models on huge data sets (up to hundreds of billions of words).

#ASTD
file_name = 'E:\\data\\cbow\\ASTD-unbalanced-not-linked.csv'
max_sent_len = 40
#ATT
file_name = 'E:\\data\\cbow\\ATT-unbalanced-not-linked.csv'
max_sent_len = 568
#HTL
file_name = 'E:\\data\\cbow\\HTL-unbalanced-not-linked.csv'
max_sent_len = 1110
##MOV
file_name = 'E:\\data\\cbow\\MOV-unbalanced-not-linked.csv'
max_sent_len = 2335
#max_sent_len = 1000
##PROD
file_name = 'E:\\data\\cbow\\PROD-unbalanced-not-linked.csv'
max_sent_len = 234
##RES
file_name = 'E:\\data\\cbow\\RES-unbalanced-not-linked.csv'
max_sent_len = 539
##LABR
file_name = 'E:\\data\\cbow\\LABR-unbalanced-not-linked-800.csv'
max_sent_len = 882




# maximum length of a sentence
print ("max_sent_len = " ,max_sent_len)


# Reading csv data
# ==================================================
#
print ("Reading text data for classification and building representations...")
reviews = []
reviews = [ ( row["text"] , row["polarity"]  ) for row in csv.DictReader(open(file_name, encoding="utf8"), delimiter=',', quoting=csv.QUOTE_NONE) ]

#Remove duplication
data = pd.DataFrame(reviews).drop_duplicates()


from LoadDataset_General import *
from Lexicon_Generation import *
import codecs
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from qalsadi import analex
from Classifiers import *
from Feature_Generation import *
import pyarabic.arabrepr
from tashaphyne.stemming import ArabicLightStemmer
from pyarabic.named import *

LoadDataset_General = LoadDataset_General()
############### Preprocessing ########
for i in range(0,len(data)):
    data.iloc[i,0] = LoadDataset_General.Emoticon_detection(data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.clean_raw_review(data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.normalizeArabic(data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.Elong_remove(data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.deNoise(data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.Remove_Stopwords(data.iloc[i,0])
    data.iloc[i,0] = LoadDataset_General.Named_Entity_Recognition(data.iloc[i,0])
#    data[i] = LoadDataset_General.Stem_word(data[i])
#    data.iloc[i,0] = LoadDataset_General.Light_Stem_word(data.iloc[i,0])
#    data[i] = LoadDataset_General.Get_root_word(data[i])

data[0][2] = LoadDataset_General.Emoticon_detection(data[0][2])


random.shuffle( data )
train_size = int(len(data) * val_split)
train_texts = data[0:train_size][0].tolist()
test_texts = data[train_size:-1][0].tolist()
train_labels = data[0:train_size][1].tolist()
test_labels = data[train_size:-1][1].tolist()
num_classes = len( set( train_labels + test_labels ) )
tokenizer = Tokenizer(nb_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ")
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
#Encode labels with value between 0 and n_class-1
le.fit( train_labels + test_labels )
train_labels = le.transform( train_labels )
test_labels = le.transform( test_labels )
print ("Classes : " + repr( le.classes_ ))

 
# CNN
# ===============================================================================================
#
print ("Method = CNN for Arabic Sentiment Analysis'")
model_variation = 'CNN-non-static'
np.random.seed(0)
nb_filter = embeddings_dim
#model = Graph()
#model.add_input(name='input', input_shape=(max_sent_len,), dtype=int)
#model.add_node(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] ), name='embedding', input='input')
#model.add_node(Dropout(dropout_prob[0]), name='dropout_embedding', input='embedding')
#for n_gram in filter_sizes:
#    model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=n_gram, border_mode='valid', activation='relu', subsample_length=1, input_dim=embeddings_dim, input_length=max_sent_len), name='conv_' + str(n_gram), input='dropout_embedding')
#    model.add_node(MaxPooling1D(pool_length=max_sent_len - n_gram + 1), name='maxpool_' + str(n_gram), input='conv_' + str(n_gram))
#    model.add_node(Flatten(), name='flat_' + str(n_gram), input='maxpool_' + str(n_gram))
#model.add_node(Dropout(dropout_prob[1]), name='dropout', inputs=['flat_' + str(n) for n in filter_sizes])
#model.add_node(Dense(1, input_dim=nb_filter * len(filter_sizes)), name='dense', input='dropout')
#model.add_node(Activation('sigmoid'), name='sigmoid', input='dense')
#model.add_output(name='output', input='sigmoid')


#model = Sequential()
main_input = Input(shape=(max_sent_len,))
embedding = Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] )(main_input)
Drop1 = Dropout(dropout_prob[0])(embedding)
i=0
conv_name=["" for x in range(len(filter_sizes))]
pool_name=["" for x in range(len(filter_sizes))]
flat_name=["" for x in range(len(filter_sizes))]
for n_gram in filter_sizes:
    conv_name[i] = str('conv_' + str(n_gram))
    conv_name[i] = Convolution1D(nb_filter=nb_filter, filter_length=n_gram, border_mode='valid', activation='relu', subsample_length=1, input_dim=embeddings_dim, input_length=max_sent_len)(Drop1)
    pool_name[i] = str('maxpool_' + str(n_gram))
    pool_name[i] = MaxPooling1D(pool_length=max_sent_len - n_gram + 1)(conv_name[i])
    flat_name[i] = str('flat_' + str(n_gram))
    flat_name[i] = Flatten()(pool_name[i])
    i+=1
#flat_name = np.array(flat_name)
merged = merge([flat_name[0], flat_name[1], flat_name[2]], mode='concat')    
droput_final = Dropout(dropout_prob[1])(merged)
Dense_final = Dense(1, input_dim=nb_filter * len(filter_sizes))(droput_final)
Out = Dense(1, activation = 'sigmoid')(Dense_final)
#model.add_output(name='output', input='sigmoid')
model = Model(inputs=main_input, outputs=Out)


# Print model summary
# ==================================================
# 
print(model.summary())

import datetime
datetime.datetime.now()
print(datetime.datetime.now())

from time import gmtime, strftime
strftime("%Y-%m-%d %H:%M:%S", gmtime())


loss_history = hist.history
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

with open('E:\\data\\Results.txt', 'a') as the_file:
    the_file.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    the_file.write('\n')
    the_file.write(model.summary())
    the_file.write('\n')


#Visualize the model in a graph
#1st method
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import vis_utils
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
#2nd method
plot_model(model, to_file='E:/model_plot.png', show_shapes=True, show_layer_names=True)

# model compilation
# ==================================================
# 
#if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='Adagrad') 
#else: model.compile(loss='categorical_crossentropy', optimizer='Adagrad')
if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy']) 
else: model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])


#  model early_stopping and checkpointer
# ==================================================
# 
Round='round-1'
Vectors = 'Mine-vec'
Linked = 'Not-Linked'
early_stopping = EarlyStopping(patience=20, verbose=1)
#Stop training when a monitored quantity has stopped improving.  
#patience: number of epochs with no improvement after which training will be stopped.
checkpointer = ModelCheckpoint(filepath= 'E:\\data\\cbow\\Weights\\'+'Mov'+'_'+Round+'_'+Linked+'_'+model_variation+'_'+Vectors+'_weights_Ar_best.hdf5', verbose=1, save_best_only=False)
#Save the model after every epoch.
checkpointer = ModelCheckpoint(filepath= 'E:\\data\\cbow\\Weights\\checkpoint_weights_Ar_best.hdf5', verbose=1, save_best_only=False)
  
#  model history
# ==================================================
# 
#  hist = model.fit({'input': train_sequences, 'output': train_labels}, batch_size=32, nb_epoch=30, verbose=2, show_accuracy=True,callbacks=[early_stopping, checkpointer])
#hist = model.fit(train_sequences, train_labels, batch_size=32, epochs=30, verbose=2, callbacks=[early_stopping, checkpointer])
hist = model.fit(train_sequences, train_labels, batch_size=32, epochs=5, verbose=2, validation_data=(test_sequences, test_labels), callbacks=[early_stopping, checkpointer])
#without chepointer
hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2, callbacks= [early_stopping])
hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=5, verbose=2, callbacks= [early_stopping])
#without early stoping
hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2)
hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=2, verbose=2)
#with only checkpoint
hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=2, verbose=2, callbacks= [checkpointer])


#1 -- fakes
#hist_log = hist.history
#dataset_name='MOV'
#numpy_loss_history = np.array(hist_log).tofile('E:\\data\\Results'+dataset_name+'.txt',sep='\n')
##np.savetxt('E:\\data\\Results.txt', numpy_loss_history, delimiter="\n")

#2--fakes
#import pickle
#with open('E:\\data\\Results.txt', 'a') as file_pi:
#        pickle.dump(hist.history, file_pi)

#3
pd.DataFrame(model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=2, verbose=2).history).to_csv("E:\\data\\historyMOV.csv")

#4
#from keras.callbacks import CSVLogger
#keras.callbacks.CSVLogger('E:\\data\\Results.txt', separator='\n', append=True)
#csv_logger = CSVLogger('E:\\data\\Results.csv')
#hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=1, verbose=2, callbacks=[csv_logger])

#5
#import sys
#oldStdout = sys.stdout
#file = open('logFile', 'w')
#sys.stdout = file
#model.fit(Xtrain, Ytrain)
#sys.stdout = oldStdout

#6
with open('E:\\data\\Results.txt', 'a') as the_file:
    the_file.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    the_file.write('\n')
    the_file.write(hist_log.)
    the_file.write('\n')




# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Evaluate model keras
# ==================================================
#
print("Evaluate...")
#  score, acc = model.evaluate({'input': test_sequences, 'output': test_labels},show_accuracy=True,batch_size=32)
score, acc  = model.evaluate(test_sequences, test_labels,batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)


# Evaluate model sklearn
# ==================================================
#
#results = np.array(model.predict({'input': test_sequences}, batch_size=32)['output'])
results = np.array(model.predict(test_sequences, batch_size=32))
if num_classes != 2: results = results.argmax(axis=-1)
else: results = (results > 0.5).astype('int32')
print ("Accuracy-sklearn = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print ('RMSE',np.sqrt(((results - test_labels) ** 2).mean(axis=0)).mean() ) # Printing RMSE
print (sklearn.metrics.classification_report( test_labels , results ))




###Here is how determinate a number of shapes of you Keras model (var model), and each shape unit occupies 4 bytes in memory:
#shapes_count = int(numpy.sum([numpy.prod(numpy.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))
#memory = shapes_count * 4
####And here is how determinate a number of params of you Keras model (var model):
#from keras import backend as K
#trainable_count = int(numpy.sum([K.count_params(p) for p in set(model.trainable_weights)]))
#non_trainable_count = int(numpy.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
#####the memory required for various batch sizes?
# of parameters * 4 bits + size of batch * some constant.