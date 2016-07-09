#!/usr/bin/env
from operator import itemgetter
import pickle
import numpy as np
import sys
sys.path.append('pre_processing')
import clean_text as ct
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.classifier import Classifier
import json



def eval(path):
    pkl_file = open('train.pkl', 'rb')
    authorsTrainer_data = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file2 = open('js.pkl', 'rb')
    js = pickle.load(pkl_file2)
    pkl_file2.close()

    authorsClassifier = Classifier(authorsTrainer_data, tokenizer)


    input = list(open(path).readlines())
    res = ""
    for t in input:
        t = ct.stopping(t)
        t = ct.stemming(t)
        res += t.replace(",","").replace(";","").replace(")","").replace("(","").replace(".","").replace(":","").replace("!","").replace("?","")



    classification = authorsClassifier.classify(res)


    classification = sorted(classification, key=itemgetter(1), reverse=True)


    vec=[]
    ALOT = 1.79769313e+308
    sum = 0.0
    for p in classification:
        sum += p[1]
    sum = max(min(sum, ALOT), -ALOT)

    prova=[]

    for p in classification:
        val = p[1]
        prova.append(val)
    maximum = max(prova)
    dict = []
    for p in classification:
        #curr=""
        # for d in js:
        #     if d['file_name']==p[0]+".txt":
        #         curr = d['file_name']
        #         lung=len(open("data/input_stemmed/"+curr).readlines())
        # val = ((p[1])/sum)*100
        # prova.append(val)
        val = p[1]
        #100 oppure ALOT
        #print p[0],"===========>", val/maximum if val < np.inf else 100
        dict.append((str(p[0])+'.txt', str(val/maximum)))

    for d in dict:
        print d[0], d[1]
    with open('data/authors.json') as authors_json:
        authors = json.load(authors_json)
    for author in authors:
        for d in dict:
            if d[0] == author['file_name']:
                vec.append(float(d[1]))
                print "AUTORE", d[0], "VALORE", d[1]
    return vec

