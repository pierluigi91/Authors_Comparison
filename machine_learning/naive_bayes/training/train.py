#!/usr/bin/env
"""
Suppose you have some texts of news and know their categories.
You want to train a system with this pre-categorized/pre-classified
texts. So, you have better call this data your training set.
"""
import pickle
import sys
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
import json


def train():
    authorsTrainer = Trainer(tokenizer)


    autori= []
    length_opere=[]
    with open('data/authors.json') as data_file:
        js = json.load(data_file)
    for d in js:
        opera = open('data/stemmed_data/'+d['file_name'], "r").readlines()  #CAMBIARE QUI PER LE OPERE
        length_opere.append(len(opera))
        r = ""
        for p in opera:
            r += p
        autori.append(r)
    #max_length = float(max(length_opere))


    authorsSet =[
        {'text': autori[0], 'author': 'arthur_conan_doyle'},
        {'text': autori[1], 'author': 'charles_darwin'},
        {'text': autori[2], 'author': 'charles_dickens'},
        {'text': autori[3], 'author': 'daniel_defoe'},
        {'text': autori[4], 'author': 'howard_phillips_lovecraft'},
        {'text': autori[5], 'author': 'james_joyce'},
        {'text': autori[6], 'author': 'jane_austen'},
        {'text': autori[7], 'author': 'jonathan_swift'},
        {'text': autori[8], 'author': 'mark_twain'},
        {'text': autori[9], 'author': 'mary_shelley'},
        {'text': autori[10], 'author': 'oscar_wilde'},
        {'text': autori[11], 'author': 'robert_louis_stevenson'},
        {'text': autori[12], 'author': 'virginia_woolf'}

    ]

    lista_num_token=[]
    for author in authorsSet:
        lista_num_token.append(authorsTrainer.min_num_token(author['text']))
    min_num_token = min(lista_num_token)



    for author in authorsSet:
        authorsTrainer.train(author['text'], author['author'], min_num_token)






    #SERVE SOLO A NAIVE BAYES
    output = open('train.pkl', 'wb')
    pickle.dump(authorsTrainer.data, output)
    output.close()

    #SERVE ANCHEA DEEP LEARNING
    output2 = open('js.pkl', 'wb')
    pickle.dump(js, output2)
    output2.close()

