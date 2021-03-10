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
from LoadDataset_General import *
#LoadDataset_HTLs = LoadDataset_HTLs()
from nltk import bigrams
from nltk import trigrams

class Lexicon_Generation:
    def __init__(self):
        self.methods = ('Lexicons')  
        self.lexicon_path="E:\\data\\Lexicons\\"
        self.lexicon_path_lex="E:\\data\\Lexicons\\dr_samha_lex\\"
        
    def getWordLists(self):
#https://github.com/alexrutherford/arabic_nlp/blob/master/get_sentiment.py
        import LoadDataset_General
        '''Reads terms to be tsted against text from files. Returns tuple of lists of words.'''
#        posWords=pd.read_csv(self.lexicon_path+'pos_words.txt')
#        negWords=pd.read_csv(self.lexicon_path+'neg_words_all.txt')
        posWords=pd.read_csv(self.lexicon_path_lex+'Pos.txt')
        negWords=pd.read_csv(self.lexicon_path_lex+'Neg.txt')
#        negFileAdd=pd.read_csv(lexicon_path+'neg_words_all.txt')
        negationWords=pd.read_csv(self.lexicon_path+'negation_words.txt')
        posEmojis=pd.read_csv(self.lexicon_path+'pos_emojis.txt')
        negEmojis=pd.read_csv(self.lexicon_path+'neg_emojis.txt')
        posWords = posWords.iloc[:,0].values.tolist()
        negWords = negWords.iloc[:,0].values.tolist()
        for i in range(0,len(posWords)):
            posWords[i] = LoadDataset_General.LoadDataset_General.normalizeArabic(LoadDataset_General.LoadDataset_General(),posWords[i])
            posWords[i] = LoadDataset_General.LoadDataset_General.Elong_remove(LoadDataset_General.LoadDataset_General(),posWords[i])
            posWords[i] = LoadDataset_General.LoadDataset_General.Light_Stem_word(LoadDataset_General.LoadDataset_General(),posWords[i])
        for i in range(0,len(negWords)):
            negWords[i] = LoadDataset_General.LoadDataset_General.normalizeArabic(LoadDataset_General.LoadDataset_General(),negWords[i])
            negWords[i] = LoadDataset_General.LoadDataset_General.Elong_remove(LoadDataset_General.LoadDataset_General(),negWords[i])
            negWords[i] = LoadDataset_General.LoadDataset_General.Light_Stem_word(LoadDataset_General.LoadDataset_General(),negWords[i])
        posWords = [x for i, x in enumerate(posWords) if i == posWords.index(x)]
        negWords = [x for i, x in enumerate(negWords) if i == negWords.index(x)]
        return posWords,negWords,negationWords,posEmojis,negEmojis

    def getHTLLexicon(self):
        HTL_lex = pd.read_csv('E:\\data\\Large_SA\\HTL_lex.csv')
        
        return HTL_lex
        
    def pos_neg_counts(self, LoadDataset_General, body, posWords, negWords):
        posCount_1 = 0
        negCount_1 = 0
        posCount_2 = 0
        negCount_2 = 0
        posCount_3 = 0
        negCount_3 = 0
        posCount = 0
        negCount = 0
        
        word = body.split(u" ")
        word_final = list()
        posCount_1 = 0
        negCount_1 = 0
        for w in word:
            if w in posWords:
                posCount_1+=1
            if w in negWords:
                negCount_1+=1

#        word = [x for x in re.split(u'(.*?\s.*?)\s', body_new) if x]
        bigram_body=list(bigrams(body.split()))
        word = [x[0]+" "+x[1] for x in bigram_body]
        word_final = list()
        posCount_2 = 0
        negCount_2 = 0
        for w in word:
            if w in posWords:
                posCount_2+=1
            if w in negWords:
                negCount_2+=1
#        word = [x for x in re.split(u'(.*?\s.*?\s.*?)\s', body_new) if x]
        trigram_body=list(trigrams(body.split()))
        word = [x[0]+" "+x[1]+" "+x[2] for x in trigram_body]
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
        return posCount, negCount

    def pos_neg_Emojiss(self, LoadDataset_General, body, posEmojis, negEmojis):
        ############ Pos and Neg Emojis ######
        posEmojis_Count = 0
        negEmojis_Count = 0
        word_emj = body.split(u" ")
        for w in word_emj:
            if w in posEmojis:
                posEmojis_Count+=1
            elif w in negEmojis:
                negEmojis_Count+=1
        return posEmojis_Count, negEmojis_Count
                    
    def Length_Review(self, LoadDataset_General, body):
        words = body.split(u" ")
        length = len(words)
        return length
    
    def Lexicon_Sentiment(self, body, posWords, negWords):
        posSent_col = np.ones(len(posWords), dtype=int).reshape(-1,1)
        negSent_col = np.full(len(negWords), -1, dtype=int).reshape(-1,1)
        posWords = np.c_[posWords, posSent_col]
        negWords = np.c_[negWords, negSent_col]
        Lexicon = pd.DataFrame(np.vstack((posWords,negWords)))
        ##Remove deuplication
        Lexicon = pd.DataFrame(Lexicon.drop_duplicates().values)
        
        Lex_Words = Lexicon.iloc[:,0].values.tolist()
        Lex_Words_S = [int(a) for a in Lexicon.iloc[:,1].values.tolist()]
        SentCount = 0
        ############ Pos and Neg Words ######
        word = body.split(u" ")
        SentCount = 0
        for w in word:
            if w in Lex_Words:
                SentCount+=Lex_Words_S[Lex_Words.index(w)]
    
        bigram_body=list(bigrams(body.split()))
        word = [x[0]+" "+x[1] for x in bigram_body]    
        for w in word:
            if w in Lex_Words:
                SentCount+=Lex_Words_S[Lex_Words.index(w)]
                
        trigram_body=list(trigrams(body.split()))
        word = [x[0]+" "+x[1] for x in trigram_body]
        for w in word:
            if w in Lex_Words:
                SentCount+=Lex_Words_S[Lex_Words.index(w)]
                    
        return SentCount
    

    