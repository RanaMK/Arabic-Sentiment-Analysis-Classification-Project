# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:21:30 2017
@author: PAVILION G4
"""
import codecs
import numpy as np
import pandas as pd
import re
from LoadDataset_BBN import *
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
LoadDataset_BBN=LoadDataset_BBN()
(body,rating)=LoadDataset_BBN.Load_data()

############ Preprocessing ########
for i in range(0,len(body)):
    body[i] = LoadDataset_BBN.clean_raw_review(body[i])
    body[i] = LoadDataset_BBN.normalizeArabic(body[i])
    body[i] = LoadDataset_BBN.Elong_remove(body[i])
    body[i] = LoadDataset_BBN.deNoise(body[i])
    body[i] = LoadDataset_BBN.Named_Entity_Recognition(body[i])
#    body[i] = LoadDataset_BBN.Stem_word(body[i])
    body[i] = LoadDataset_BBN.Light_Stem_word(body[i])
#    body[i] = LoadDataset_BBN.Get_root_word(body[i])

#### Load unbalanced dataset
(unbalanced_train_x, unbalanced_train_y, unbalanced_test_x, unbalanced_test_y, unbalanced_valid_x, unbalanced_valid_y) = LoadDataset_BBN.get_train_test_validation_unbalanced(body, rating)
d_train = np.concatenate((unbalanced_train_x, unbalanced_valid_x))
Y_train = np.concatenate((unbalanced_train_y, unbalanced_valid_y))
Y_test = unbalanced_test_y

########## Feature Selection ############
###### Counts ##########
# tokenizer
an = analex.analex()
tokenizer = an.text_tokenize
tf_fg = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 1))
#tf_fg = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))
#tf_fg = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 3))

##### tfidf ############
# tokenizer
an = analex.analex()
tokenizer = an.text_tokenize
tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 1))
#tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 2))
#tf_fg = TfidfVectorizer(tokenizer=an.text_tokenize, ngram_range=(1, 3))

Features = tf_fg.fit_transform(Lex_Words)
X_train = tf_fg.transform(d_train)
X_test = tf_fg.transform(unbalanced_test_x)
tf_fg.get_feature_names()

############# Features using counts and tfidf ######
X_train = tf_fg.fit_transform(d_train)
X_test = tf_fg.transform(unbalanced_test_x)

### POC for tfidf
#count_words = 0
#all_words = list()
#for i in range(0,len(d_train)):
#    word = d_train[i].split(u" ")
#    for w in word:
#        all_words.append(w)
##    count_words += len(w)
#all_words_nodups = [x for i, x in enumerate(all_words) if i == all_words.index(x)]


###### Variance Threshold
#from sklearn.feature_selection import VarianceThreshold
#sel = VarianceThreshold(threshold=(.7 * (1 - .7)))
#X_train = sel.fit_transform(X_train)
#X_test = sel.fit_transform(X_test)

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(d_train, Y_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

#### Generate X_train using Lexicons ############
Feature_Generation = Feature_Generation()
body_final = list()
posCount_v = list()
negCount_v = list()
posEmojis_Count_v = list()
negEmojis_Count_v = list()
for i in range(0,len(d_train)):
    (lexicon, posCount, negCount, posEmojis_Count, negEmojis_Count, body_txt) = Feature_Generation.Lexicon_generation(LoadDataset_BBN,d_train[i])
    body_final.append(body_txt)
    posCount_v.append(posCount)
    negCount_v.append(negCount)
    posEmojis_Count_v.append(posEmojis_Count)
    negEmojis_Count_v.append(posEmojis_Count)
### X_train
X_train= np.c_[posCount_v, negCount_v, posEmojis_Count_v, negEmojis_Count_v]

#### X_test using Lexicons ############
Feature_Generation = Feature_Generation()
body_final_t = list()
posCount_t_v = list()
negCount_t_v = list()
posEmojis_Count_t_v = list()
negEmojis_Count_t_v = list()
for i in range(0,len(d_train)):
    (lexicon, posCount_t, negCount_t, posEmojis_Count_t, negEmojis_Count_t, body_txt_t) = Feature_Generation.Lexicon_generation(LoadDataset_BBN,d_train[i])
    body_final_t.append(body_txt_t)
    posCount_t_v.append(posCount_t)
    negCount_t_v.append(negCount_t)
    posEmojis_Count_t_v.append(posEmojis_Count_t)
    negEmojis_Count_v.append(posEmojis_Count_t)
### X_test
X_test= np.c_[posCount_t_v, negCount_t_v, posEmojis_Count_t_v, negEmojis_Count_t_v]

 
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
posWords = posWords.iloc[:,0].values.tolist()
negWords = negWords.iloc[:,0].values.tolist()
for i in range(0,len(posWords)):
    posWords[i] = LoadDataset_BBN.normalizeArabic(posWords[i])
    posWords[i] = LoadDataset_BBN.Elong_remove(posWords[i])
    posWords[i] = LoadDataset_BBN.Light_Stem_word(posWords[i])
for i in range(0,len(negWords)):
    negWords[i] = LoadDataset_BBN.normalizeArabic(negWords[i])
    negWords[i] = LoadDataset_BBN.Elong_remove(negWords[i])
    negWords[i] = LoadDataset_BBN.Light_Stem_word(negWords[i])

posWords = [x for i, x in enumerate(posWords) if i == posWords.index(x)]
negWords = [x for i, x in enumerate(negWords) if i == negWords.index(x)]
lexicon = list()
lexicon = posWords + negWords
#        lexicon = posWords.append(negWords)
lexicon = [x for i, x in enumerate(lexicon) if i == lexicon.index(x)]


#### on X_train
posCount_1 = 0
negCount_1 = 0
posCount_2 = 0
negCount_2 = 0
posCount_3 = 0
negCount_3 = 0
detected_posW = list()
detected_negW = list()
posCount_new=0
negCount_new=0
body_final = list()
posCount_v = list()
negCount_v = list()
posEmojis_Count_v = list()
negEmojis_Count_v = list()
#### Remove stop words #######
for i in range(0,len(d_train)):
    word = d_train[i].split(u" ")
    word_final = list()
    posCount_1 = 0
    negCount_1 = 0
    for w in word:
        if w in stopWords:
            word_final.append(u"")
        else:
            ############ Pos and Neg Words ######
            if w in posWords:
                posCount_1+=1
            if w in negWords:
                negCount_1+=1
            word_final.append(w)
    body_new = " ".join(word_final) 
#    body_final.append(body_new) 
    
############ Pos and Neg Words ######
#for i in range(0,len(body_final)):
    word = [x for x in re.split(u'(.*?\s.*?)\s', body_new) if x]
    word_final = list()
    posCount_2 = 0
    negCount_2 = 0
    for w in word:
        if w in posWords:
            posCount_2+=1
        if w in negWords:
            negCount_2+=1
    word = [x for x in re.split(u'(.*?\s.*?\s.*?)\s', body_new) if x]
    word_final = list()
    posCount_3 = 0
    negCount_3 = 0
    for w in word:
        if w in posWords:
            posCount_3+=1
        if w in negWords:
            negCount_3+=1
    posCount = posCount_1 + posCount_2 + posCount_3
    negCount = negCount_1 + negCount_2 + negCount_3
    posCount_v.append(posCount)
    negCount_v.append(negCount)
           
############ Pos and Neg Emojis ######
    posEmojis_Count = 0
    negEmojis_Count = 0
    word_emj = body_new.split(u" ")
    word_list_emoj = list()
    for w in word_emj:
        if w in posEmojis:
            posEmojis_Count+=1
            word_list_emoj.append("ايموشنموجب")
        elif w in negEmojis:
            negEmojis_Count+=1
            word_list_emoj.append("ايموشنسالب")
        else:
            word_list_emoj.append(w)    
#      body_final[i] = " ".join(word_list_emoj) 
    body_final.append(" ".join(word_list_emoj))
    posEmojis_Count_v.append(posEmojis_Count)
    negEmojis_Count_v.append(negEmojis_Count)
    

#### on X_test
posCount_t_1 = 0
negCount_t_1 = 0
posCount_t_2 = 0
negCount_t_2 = 0
posCount_t_3 = 0
negCount_t_3 = 0
body_final_t = list()
posCount_t_v = list()
negCount_t_v = list()
posEmojis_Count_t_v = list()
negEmojis_Count_t_v = list()
#### Remove stop words #######
for i in range(0,len(unbalanced_test_x)):
    word_t = unbalanced_test_x[i].split(u" ")
    word_t_final = list()
    posCount_t_1 = 0
    negCount_t_1 = 0
    for w in word_t:
        if w in stopWords:
            word_t_final.append(u"")
        else:
            ############ Pos and Neg Words ######
            if w in posWords:
                posCount_t_1+=1
            if w in negWords:
                negCount_t_1+=1
            word_t_final.append(w)
    body_new = " ".join(word_t_final) 
#    body_final_t.append(body_new) 
        
############ Pos and Neg Words ######
#for i in range(0,len(body_final_t)):
    word_t = [x for x in re.split(u'(.*?\s.*?)\s', body_new) if x]
    posCount_t_2 = 0
    negCount_t_2 = 0
    for w in word_t:
        if w in posWords:
            posCount_t_2+=1
        if w in negWords:
            negCount_t_2+=1
    word = [x for x in re.split(u'(.*?\s.*?\s.*?)\s', body_new) if x]
    posCount_t_3 = 0
    negCount_t_3 = 0
    for w in word:
        if w in posWords:
            posCount_t_3+=1
        if w in negWords:
            negCount_t_3+=1
    posCount_t = posCount_t_1 + posCount_t_2 + posCount_t_3
    negCount_t = negCount_t_1 + negCount_t_2 + negCount_t_3
    posCount_t_v.append(posCount_t)
    negCount_t_v.append(negCount_t)
           
############ Pos and Neg Emojis ######
    posEmojis_Count_t = 0
    negEmojis_Count_t = 0
    word = body_new.split(u" ")
    word_list = list()
    for w in word:
        if w in posEmojis:
            posEmojis_Count_t+=1
            word_list.append("ايموشنموجب")
        elif w in negEmojis:
            negEmojis_Count_t+=1
            word_list.append("ايموشنسالب")
        else:
            word_list.append(w)    
    #      body_final = " ".join(word_list_emoj) 
    body_final.append(" ".join(word_list_emoj))
    posEmojis_Count_t_v.append(posEmojis_Count_t)
    negEmojis_Count_t_v.append(negEmojis_Count_t)


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
    Lexicon[0][i] = LoadDataset_BBN.normalizeArabic(Lexicon[0][i])
    Lexicon[0][i] = LoadDataset_BBN.Elong_remove(Lexicon[0][i])
    Lexicon[0][i] = LoadDataset_BBN.Light_Stem_word(Lexicon[0][i])   
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