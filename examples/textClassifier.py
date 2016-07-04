
"""
Suppose you have some texts of news and know their categories.
You want to train a system with this pre-categorized/pre-classified
texts. So, you have better call this data your training set.
"""
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages/naiveBayesClassifier")
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
from operator import itemgetter
import json
sys.path.append('/home/pierluigi/PycharmProjects/Authors_Comparison/cnn-text-classification/pre_processing')
import clean_text
import numpy as np


authorsTrainer = Trainer(tokenizer)


js = None
autori= []
length_opere=[]
with open('/home/pierluigi/PycharmProjects/Authors_Comparison/cnn-text-classification/data/authors.json') as data_file:
    js = json.load(data_file)
for d in js:
    opera = open('/home/pierluigi/PycharmProjects/Authors_Comparison/cnn-text-classification/data/input_backup/'+d['file_name'], "r").readlines()  #CAMBIARE QUI PER LE OPERE
    length_opere.append(len(opera))
    r = ""
    for p in opera:
        r += p
    autori.append(r)
max_length = float(max(length_opere))


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



newsClassifier = Classifier(authorsTrainer.data, tokenizer)


test = list(open("/home/pierluigi/Scrivania/testi/prove/hp_01.txt").readlines())
res = ""
for t in test:
    t = clean_text.stopping(t)
    t = clean_text.stemming(t)
    res += t.replace(",","").replace(";","").replace(")","").replace("(","").replace(".","").replace(":","")



classification = newsClassifier.classify(res)


classification = sorted(classification, key=itemgetter(1), reverse=True)


ALOT = 1.79769313e+308
sum = 0.0
for p in classification:
    sum += p[1]
sum = max(min(sum,ALOT),-ALOT)

prova=[]
for p in classification:
    curr=""
    for d in js:
        if d['file_name']==p[0]+".txt":
            curr = d['file_name']
            lung=len(open("/home/pierluigi/PycharmProjects/Authors_Comparison/cnn-text-classification/data/input_backup/"+curr).readlines())
    val = ((p[1])/sum)*100
    prova.append(val)
    #100 oppure ALOT
    print p[0],"===========>", val if val<np.inf else 100, "%"

# prova_np = np.asarray(prova)
# np_minmax = (prova_np - prova_np.min()) / (prova_np.max() - prova_np.min())
# for p in np_minmax:
#     print p