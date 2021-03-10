##### Lexicons code


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
Lex_Words = np.vstack((posWords,negWords))

#for i in range(0,len(posWords)):
#    posWords[i] = LoadDataset_HTLs.normalizeArabic(posWords[i])
#    posWords[i] = LoadDataset_HTLs.Elong_remove(posWords[i])
#    posWords[i] = LoadDataset_HTLs.Light_Stem_word(posWords[i])
#for i in range(0,len(negWords)):
#    negWords[i] = LoadDataset_HTLs.normalizeArabic(negWords[i])
#    negWords[i] = LoadDataset_HTLs.Elong_remove(negWords[i])
#    negWords[i] = LoadDataset_HTLs.Light_Stem_word(negWords[i])
   
for i in range(0,len(Lex_Words)):
    Lex_Words[i,0] = LoadDataset_HTLs.normalizeArabic(Lex_Words[i,0])
    Lex_Words[i,0] = LoadDataset_HTLs.Elong_remove(Lex_Words[i,0])
    Lex_Words[i,0] = LoadDataset_HTLs.Light_Stem_word(Lex_Words[i,0])    

posWords = [x for i, x in enumerate(posWords) if i == posWords.index(x)]
negWords = [x for i, x in enumerate(negWords) if i == negWords.index(x)]
lexicon = list()
lexicon = posWords + negWords
#        lexicon = posWords.append(negWords)
lexicon = [x for i, x in enumerate(lexicon) if i == lexicon.index(x)]
Lex_Words = [x for i, x in enumerate(Lex_Words) if i == Lex_Words.index(x)]

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

###### HTL Lexicon
########## HTL Lex ###########
HTL_lex = pd.read_csv('E:\\data\\Large_SA\\HTL_lex.csv')
ALL_lex = pd.read_csv('E:\\data\\Large_SA\\ALL_lex.csv')
HTL_words = HTL_lex.iloc[:,0].values.tolist()
HTL_words_S = HTL_lex.iloc[:,1].values.tolist()
ALL_words = ALL_lex.iloc[:,0].values.tolist()
ALL_words_S = ALL_lex.iloc[:,1].values.tolist()
for i in range(0,len(ALL_words)):
    ALL_words[i] = LoadDataset_HTLs.clean_raw_review(ALL_words[i])
    ALL_words[i] = LoadDataset_HTLs.normalizeArabic(ALL_words[i])
    ALL_words[i] = LoadDataset_HTLs.Elong_remove(ALL_words[i])
    ALL_words[i] = LoadDataset_HTLs.deNoise(ALL_words[i])
#    body[i] = LoadDataset_HTLs.Named_Entity_Recognition(body[i])
#    body[i] = LoadDataset_HTLs.Stem_word(body[i])
    ALL_words[i] = LoadDataset_HTLs.Light_Stem_word(ALL_words[i])
#    body[i] = LoadDataset_HTLs.Get_root_word(body[i])

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
        if w in ALL_words:
            SentCount+=ALL_words_S[ALL_words.index(w)]

#    word = [x for x in re.split(u'(.*?\s.*?)\s', body[i]) if x]
    bigram_body=list(bigrams(body[i].split()))
#    new = [item for sub in bi_body0 for item in sub]
    word = [x[0]+" "+x[1] for x in bigram_body]    
    for w in word:
        if w in ALL_words:
            SentCount+=ALL_words_S[ALL_words.index(w)]
            
#    word = [x for x in re.split(u'(.*?\s.*?\s.*?)\s', body[i]) if x]
    trigram_body=list(trigrams(body[i].split()))
#    new = [item for sub in bi_body0 for item in sub]
    word = [x[0]+" "+x[1] for x in trigram_body]
    for w in word:
        if w in ALL_words:
            SentCount+=ALL_words_S[ALL_words.index(w)]
            
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

