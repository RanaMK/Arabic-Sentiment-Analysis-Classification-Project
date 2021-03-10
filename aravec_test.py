# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:53:39 2018

@author: PAVILION G4
"""

# -*- coding: utf8 -*-
import gensim
import re

# load the model
fname='E:\\data\\aravec\\Twt-CBOW'
model = gensim.models.Word2Vec.load(fname)

# Clean/Normalize Arabic Text
def clean_str(text):
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']
    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    
    #remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])
    
    #trim    
    text = text.strip()

    return text

# python 3.X
word = clean_str(u'القاهرة')
# python 2.7
# word = clean_str('القاهرة'.decode('utf8', errors='ignore'))

# find and print the most similar terms to a word
most_similar = model.wv.most_similar( word )
for term, score in most_similar:
	print(term, score)
	
# get a word vector
word_vector = model.wv[ word ]

##check on a sentence

# python 3.X
import numpy as np
word1 = clean_str(u'القاهرة')
word2 = clean_str(u'سوهاج')
word = list()
word = np.append(word1, word2)
# python 2.7
# word = clean_str('القاهرة'.decode('utf8', errors='ignore'))

# find and print the most similar terms to a word
most_similar = list()
for i in range(0,2):
    most_similar.append(model.wv.most_similar( word[i] ))
#for term, score in most_similar:
#	print(term, score)
	
# get a word vector
word_vector = model.wv[ word ]

