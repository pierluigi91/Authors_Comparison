#!/usr/bin/env
from operator import itemgetter
import pickle
import sys
sys.path.append('pre_processing')
import clean_text as ct
from machine_learning.naive_bayes import tokenizer
from machine_learning.naive_bayes import classifier as Classifier
import json



def eval(path):
    pkl_file = open('train.pkl', 'rb')
    authorsTrainer_data = pickle.load(pkl_file)
    pkl_file.close()

    authorsClassifier = Classifier.Classifier(authorsTrainer_data, tokenizer)


    input = list(open(path).readlines())
    res = ""
    for t in input:
        t = ct.stopping(t)
        t = ct.stemming(t)
        res += t.replace(",","").replace(";","").replace(")","").replace("(","").replace(".","").replace(":","").replace("!","").replace("?","")

    classification = authorsClassifier.classify(res)

    classification = sorted(classification, key=itemgetter(1), reverse=True)

    vec=[]
    prova=[]

    for p in classification:
        val = p[1]
        prova.append(val)
    maximum = max(prova)
    dict = []
    for p in classification:
        val = p[1]
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

