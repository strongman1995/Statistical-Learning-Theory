
# coding: utf-8

# In[8]:

import os
import pickle
import numpy as np
import random
import math

data_dir = "hw2data/"


# In[5]:

stop_words = list()
with open(os.path.join(data_dir, "stop_words_eng.txt")) as f:
    for line in f.readlines():
        stop_words.append(line.strip())

voc = list()
with open(os.path.join(data_dir, "vocabulary.txt")) as f:
    for line in f.readlines():
        voc.append(line.strip())
        
stop_idxs = list()
for stop_w in stop_words:
    for i, v in enumerate(voc):
        if v == stop_w:
            stop_idxs.append(i+1)

# save stop words in stop_word_index.txt
with open(os.path.join(data_dir, 'stop_word_index.txt'), 'w') as f:
    for idx in stop_idxs:
        f.write(str(idx)+'\n')


# In[75]:

# import data
def import_data(filename, stop_idxs):
    """
    filename: train/test
    """
    data_filepath = os.path.join(data_dir, filename+".data")
    label_filepath = os.path.join(data_dir, filename+".label")

    doc = list()
    word = list()
    count = list()
    with open(data_filepath) as f:
        temp_data = f.readlines()
        for row in temp_data:
            doc_id, word_id, count_ = row.split(' ')
            doc.append(int(doc_id))
            word.append(int(word_id))
            count.append(int(count_))
            
    doc_num = max(doc)
    data = [dict()]*doc_num
    for i in range(len(doc)):
        # drop stop words
        if word[i] in stop_idxs:
            continue
        data[doc[i]-1][word[i]] = count[i]
    
    label = [None]*doc_num
    with open(label_filepath) as f:
        temp_label = f.readlines()
        for i, row in enumerate(temp_label):
            label[i] = int(row)
    return data, label


# In[76]:

train_X, train_y = import_data("train", stop_idxs)
test_X, test_y = import_data("test", stop_idxs)


# In[10]:

with open(os.path.join(data_dir,"train_data.pkl"), 'w') as f:
    pickle.dump(train_X,f)  
with open(os.path.join(data_dir,"test_data.pkl"), 'w') as f:
    pickle.dump(test_X,f)  
# train_X = pickle.load(open(os.path.join(data_dir, "train_data.pkl")))
# test_X = pickle.load(open(os.path.join(data_dir, "test_data.pkl")))


# In[78]:

newsgroups_train = fetch_20newsgroups(subset='train', remove=['headers', 'footers', 'quotes'])


# In[79]:

print newsgroups_train.filenames.shape, newsgroups_train.target.shape, newsgroups_train.target[:10]


# In[44]:

tf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,stop_words='english')
tf = tf_vectorizer.fit_transform(newsgroups_train.data)

# count_vectorizer = CountVectorizer(max_df=0.95, min_df=2,stop_words='english')
# cv = count_vectorizer.fit_transform(newsgroups_train.data)

target = newsgroups_train.target

data = tf.tolil()


# In[87]:

with open(os.path.join(data_dir, "stop_words_eng.txt"), "r") as inStopwords:
    linesSw = inStopwords.readlines()
mapSW = {}
for line in linesSw:
    mapSW[line.strip("\r\n")] = 1

with open(os.path.join(data_dir, "vocabulary.txt"), "r") as inVocFile:
    linesVoc = inVocFile.readlines()
nW = len(linesVoc)

wordIdMap = []
for line in linesVoc:
    wordIdMap.append(line.strip("\n"))
wordIdMap = np.array(wordIdMap)

with open(os.path.join(data_dir, "train.data"), "r") as inDataFile:
    linesData = inDataFile.readlines()
lenData = len(linesData)
nD = int(linesData[lenData-1].split()[0])

TdwMtr = np.zeros([nD, nW])


# In[88]:

# initialize TdwMtr, word count in each word of each count
for line in linesData:
    tmp = line.split()
    idD, idW, wcnt = int(tmp[0]), int(tmp[1]), int(tmp[2])
    TdwMtr[idD-1, idW-1] = wcnt


# In[89]:

# filter stopwords
filterArr = []
for i in range(nW):
    if mapSW.has_key(wordIdMap[i]):
        continue
    filterArr.append(i)
nW = len(filterArr)
TdwMtr = TdwMtr[:, filterArr]
wordIdMap = wordIdMap[filterArr]


# In[90]:

# extract TF-IDF feature
idfArr = np.ones(nW) * nD
for i in range(nW):
    idfArr[i] = math.log(idfArr[i] / (len(np.where(TdwMtr[:, i] > 0)[0]) + 1))


# In[91]:

# TdwMtr = TdwMtr * idfArr
# using TF-IDF to filter vocabulary list
nW = 800
tfArr = np.sum(TdwMtr, 0)
valArr = tfArr * idfArr
vidArr = np.argsort(valArr)
wordIdMap = wordIdMap[vidArr[-nW:]]
TdwMtr = TdwMtr[:, vidArr[-nW:]]


# In[92]:

# vary nK to get result
for nK in [5, 10, 20, 30]:
    initUwkMtr = np.random.dirichlet(np.ones(nW), nK).transpose()
    # arguments
    smooth1 = 1
    smooth2 = 1
    nIt = 15

    YdkMtr = np.zeros([nD, nK])
    UwkMtr = np.copy(initUwkMtr)
    PkArr = np.ones(nK) / nK

    for it in range(nIt):
    #     print "iteration ", it

    #     print "E-step ..."
        # E-step
    #     Wrote E-step in for loop
    #     for d in range(nD):
    # #         print "iteration", it, "E-step estimate", d+1, "document"
    #         logSumProb = 0
    #         YdkMtr[d, :] = np.log(PkArr) + np.dot(TdwMtr[d, :], np.log(UwkMtr))
    #         logSumProb = np.max(YdkMtr[d, :])
    #         logSumProb = logSumProb + math.log(np.sum(np.exp(YdkMtr[d, :] - logSumProb)))
    #         if logSumProb == 0:
    #             print "Error: logSumProb == 0 in E-step"
    #         else:
    #             YdkMtr[d, :] = np.exp(YdkMtr[d, :] - logSumProb)
        # Wrote E-step in matrix manipulation
        YdkMtr = np.dot(TdwMtr, np.log(UwkMtr))
        YdkMtr += np.log(PkArr)
        regMaxArr = np.max(YdkMtr, 1)
        regSumArr = regMaxArr + np.log(np.sum(np.exp((YdkMtr.transpose() - regMaxArr).transpose()), 1))
        YdkMtr = np.exp((YdkMtr.transpose() - regSumArr).transpose())

    #     print "M-step ..."
        # M-step
    #     Wrote in for loop
    #     for j in range(nK):
    # #         print "iteration", it, "M-step maximize", j+1, "pi argument"
    # #         sumProb = 0
    # #         for d in range(nD):
    # #            sumProb += YdkMtr[d, j]
    #         PkArr[j] = (np.sum(YdkMtr[:,j]) + smooth1) / (nD + nK * smooth1)    
        # Wrote in matrix manipulation
        PkArr = (np.sum(YdkMtr, 0) + smooth1) / (nD + nK * smooth1)

    #     Wrote in for loop
    #     for j in range(nK):
    # #         print "iteration", it, "M-step maximize", j+1, "u argument"
    #         sumProb = 0
    #         for i in range(nW):
    #             tv = smooth2
    # #             for d in range(nD):
    # #                 tv += TdwMtr[d, i] * YdkMtr[d, j]
    #             tv += np.ma.innerproduct(TdwMtr[:, i], YdkMtr[:, j])
    #             sumProb += tv
    #             UwkMtr[i, j] = tv
    #         if sumProb == 0:
    #             print "Error: sumProb == 0 in M-step"
    #         else:
    #             UwkMtr[:, j] /= sumProb
        # Wrote in matrix manipulation
        UwkMtr = np.dot(TdwMtr.transpose(), YdkMtr)
        UwkMtr += np.ones([nW, nK])*smooth2
        regSumArr = np.sum(UwkMtr, 0)
        UwkMtr = UwkMtr / regSumArr
    idK = np.argmax(YdkMtr, 1)
    print nK, "topics in total"
    for k in range(nK):
        tidArr = np.where(idK == k)[0]
        if len(tidArr) == 0:
            print "There is 0 document belong to topic", k
            continue
        ts = np.sum(TdwMtr[tidArr, :], 0)
        widMaxFreq = np.argmax(ts)
        print "The most frequent word in topic", k, 
        print "is", wordIdMap[widMaxFreq], 
        print ", relative freq:", ts[widMaxFreq]/np.sum(ts), ", freq:", ts[widMaxFreq]


# In[ ]:



