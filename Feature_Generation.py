# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:25:33 2017

@author: PAVILION G4
"""

import re
import nltk
import pandas as pd
import numpy as np
import codecs
from Load_Dataset_HTLs import *
#LoadDataset_HTLs = LoadDataset_HTLs()

class Feature_Generation:
    def __init__(self):
        self.methods = ('Lexicons')  
        self.lexicon_path="E:\\data\\Lexicons\\"
        self.lexicon_path_lex="E:\\data\\Lexicons\\dr_samha_lex\\"
        
    def getWordLists(self):
#https://github.com/alexrutherford/arabic_nlp/blob/master/get_sentiment.py
        import Load_Dataset_HTLs
        '''Reads terms to be tsted against text from files. Returns tuple of lists of words.'''
#        posWords=pd.read_csv(self.lexicon_path+'pos_words.txt')
#        negWords=pd.read_csv(self.lexicon_path+'neg_words_all.txt')
        posWords=pd.read_csv(self.lexicon_path_lex+'Pos.txt')
        negWords=pd.read_csv(self.lexicon_path_lex+'Neg.txt')
#        negFileAdd=pd.read_csv(lexicon_path+'neg_words_all.txt')
        stopWords=pd.read_csv(self.lexicon_path+'stop_words.txt')
        negationWords=pd.read_csv(self.lexicon_path+'negation_words.txt')
        posEmojis=pd.read_csv(self.lexicon_path+'pos_emojis.txt')
        negEmojis=pd.read_csv(self.lexicon_path+'neg_emojis.txt')
        posWords = posWords.iloc[:,0].values.tolist()
        negWords = negWords.iloc[:,0].values.tolist()
        for i in range(0,len(posWords)):
            posWords[i] = Load_Dataset_HTLs.LoadDataset_HTLs.normalizeArabic(Load_Dataset_HTLs.LoadDataset_HTLs(),posWords[i])
            posWords[i] = Load_Dataset_HTLs.LoadDataset_HTLs.Elong_remove(Load_Dataset_HTLs.LoadDataset_HTLs(),posWords[i])
            posWords[i] = Load_Dataset_HTLs.LoadDataset_HTLs.Light_Stem_word(Load_Dataset_HTLs.LoadDataset_HTLs(),posWords[i])
        for i in range(0,len(negWords)):
            negWords[i] = Load_Dataset_HTLs.LoadDataset_HTLs.normalizeArabic(Load_Dataset_HTLs.LoadDataset_HTLs(),negWords[i])
            negWords[i] = Load_Dataset_HTLs.LoadDataset_HTLs.Elong_remove(Load_Dataset_HTLs.LoadDataset_HTLs(),negWords[i])
            negWords[i] = Load_Dataset_HTLs.LoadDataset_HTLs.Light_Stem_word(Load_Dataset_HTLs.LoadDataset_HTLs(),negWords[i])
        posWords = [x for i, x in enumerate(posWords) if i == posWords.index(x)]
        negWords = [x for i, x in enumerate(negWords) if i == negWords.index(x)]
        lexicon = list()
        lexicon = posWords + negWords
#        lexicon = posWords.append(negWords)
        lexicon = [x for i, x in enumerate(lexicon) if i == lexicon.index(x)]
             
        return lexicon, posWords,negWords,stopWords,negationWords,posEmojis,negEmojis

    def getHTLLexicon(self):
        HTL_lex = pd.read_csv('E:\\data\\Large_SA\\HTL_lex.csv')
        
        return HTL_lex
        
    def Lexicon_generation(self, LoadDataset_HTLs, body):
        
        (lexicon, posWords,negWords,stopWords,negationWords,posEmojis,negEmojis) = self.getWordLists()
        posCount_1 = 0
        negCount_1 = 0
        posCount_2 = 0
        negCount_2 = 0
        posCount_3 = 0
        negCount_3 = 0
#        detected_posW = list()
#        detected_negW = list()
        posCount = 0
        negCount = 0
        body_final = list()
#        posCount_v = list()
#        negCount_v = list()
#        posEmojis_Count_v = list()
#        negEmojis_Count_v = list()
        #### Remove stop words #######
#    for i in range(0,len(body)):
        word = body.split(u" ")
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
#            posCount_v.append(posCount)
#            negCount_v.append(negCount)
               
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
#            posEmojis_Count_v.append(posEmojis_Count)
#            negEmojis_Count_v.append(negEmojis_Count)
                    
        ########## HTL Lex ###########
        HTL_lex = pd.read_csv('E:\\data\\Large_SA\\HTL_lex.csv')
        HTL_words = HTL_lex.iloc[:,0].values.tolist()
        lex_words = 0
        for i in (0,1):#range(0,len(body)):
            word = body[i].split(u" ")
            word_final = list()
            for w in word:
                if w in HTL_words:
                    lex_words+=1

        return lexicon, posCount, negCount, posEmojis_Count, negEmojis_Count, body_final