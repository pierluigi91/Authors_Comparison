from pyspark import SparkConf, SparkContext
import re
from scipy import spatial
import json
import numpy as np
from operator import add
from nltk.tag import pos_tag
from nltk import word_tokenize


def map_red_less_used(path, num):
    conf = (SparkConf().setMaster('local').setAppName('mapred'))
    sc = SparkContext(conf=conf)
    text = sc.textFile(path)
    mapred_01 = text.flatMap(lambda line: line.split(" ")) \
                                .map(lambda word: (word.encode("utf-8"), 1)) \
                                .reduceByKey(lambda a,b: a+b)
    mapred_02 = mapred_01.map(lambda (a,b): (b,a)).sortByKey(True, 1).take(num)

    return mapred_02


def hapax_dix_legomena(sc, path):
    text = sc.textFile(path)
    mapred_01 = text.flatMap(lambda line: re.sub(r"[^A-Za-z\s]", "", line).split(" ")) \
        .map(lambda word: (word.encode("utf-8"), 1)) \
        .reduceByKey(lambda a, b: a + b)
    unique_words = len(mapred_01.countByKey())
    mapred_02 = dict(mapred_01.map(lambda (a, b): (b, a)).countByKey().items())
    hapax = 0
    if mapred_02.get(1) is not None:
        hapax = float(mapred_02[1])/float(unique_words)
    dix = 0
    if mapred_02.get(2) is not None:
        dix = float(mapred_02[2])/float(unique_words)
    return hapax, dix


def average_words_length(sc, path):
    text = sc.textFile(path)
    mapred_01 = text.flatMap(lambda line: re.sub(r"[^A-Za-z\s]", "", line).split(" ")).map(lambda a: (a.encode("utf-8"), (len(a))))
    tot = mapred_01.values().sum()

    average_length = tot/float(mapred_01.count())
    return average_length


def average_sentences_length(sc, path):
    text = sc.textFile(path)
    mapred_01 = text.map(lambda line: len(re.sub(r"[^A-Za-z\s]", "", line).split()))
    return mapred_01.mean()


def conj_count(sc, path):
    text = sc.textFile(path, 128)
    num_sentences = text.count()
    mapred_01 = text.map(lambda line: pos_tag(word_tokenize(line))).reduce(lambda x,y:x+y)
    mapred_02 = sc.parallelize(mapred_01).map(lambda (a,b):(b,a)).reduceByKey(lambda x, y: (str(x)+" "+str(y)))
    #mapred_02.saveAsTextFile("TAGGED")
    mapred_03 = mapred_02.map(lambda (a,b): (a, len(list(b.split()))))
    #mapred_03.saveAsTextFile("COUNT")
    num_conj = mapred_03.collectAsMap().get('IN')
    return num_conj/float(num_sentences)


def is_conj(word):
    conjs = ["after", "how", "till", "'til", "although", "if", "unless", "as", "inasmuch", "until", "when", "lest",
             "whenever", "where", "wherever", "since", "while", "because", "before", "than", "that", "though"]
    if word in conjs:
        return True
    else:
        return False

def conj_count2(sc, path):
    text = sc.textFile(path)
    num_sentences = text.count()
    num_conj = text.flatMap(lambda line: re.sub(r"[^A-Za-z\s]", "", line).split(" "))\
        .map(lambda a: ("IN", 1) if is_conj(a) else ("Null", 1)).reduceByKey(lambda a, b: a+b).collectAsMap().get("IN")
    return num_conj/float(num_sentences)


def pos_tagging(path, split_size, sc): # DA FARE SUL TESTO PRE-STEMMING
    text = sc.textFile(path, split_size)
    mapred = text.flatMap(lambda line: pos_tag(line.split())).filter(lambda line: 'IN' in line)
    return float(mapred.count())/float(text.count())

def get_spark_vector(path):
    conf = (SparkConf().setMaster('local[*]').setAppName('spark'))
    sc = SparkContext(conf=conf)
    vector = list()
    print " "
    print " "
    print " "
    print " "
    print "----------------------- HAPAX DIX ------------------------"
    print " "
    print " "
    print " "
    print " "
    hapax_dix_result = hapax_dix_legomena(sc, path)
    vector.append(hapax_dix_result[0])
    vector.append(hapax_dix_result[1])
    print " "
    print " "
    print " "
    print " "
    print "----------------------- WORDS ------------------------"
    print " "
    print " "
    print " "
    print " "
    #vector.append(average_words_length(sc, path))
    print " "
    print " "
    print " "
    print " "
    print "----------------------- SENTENCES ------------------------"
    print " "
    print " "
    print " "
    print " "
    vector.append(average_sentences_length(sc, path))
    print " "
    print " "
    print " "
    print " "
    print "----------------------- CONJ ------------------------"
    print " "
    print " "
    print " "
    print " "
    vector.append(conj_count2(sc, path))
    print vector
    sc.stop()
    return vector


def train_vectors():
    with open('data/authors.json') as data_file:
        js = json.load(data_file)
    vectors_json = {}
    for d in js:
        vectors_json[str(d['file_name'])] = get_spark_vector('data/sentence_to_line_data/' + d['file_name'])

    with open('authors_vector.json', 'w') as vectors:
        json.dump(vectors_json, vectors)


def evaluate(path):
    input_vector = np.array(get_spark_vector(path))
    with open('data/authors.json') as authors_json:
        authors = json.load(authors_json)
    with open('authors_vector.json') as vectors_json:
        vectors = json.load(vectors_json)
    distance = 100
    distance_vector = list()
    for author in vectors:
        temp_dist = spatial.distance.cosine(input_vector, np.array(vectors[author]))
        for aut in authors:
            if aut['file_name'] == author:
                print "DISTANZA DA " + str(author) + " = " + str(temp_dist)
                distance_vector.insert(aut['index'], temp_dist)

    distance_vector = np.array(distance_vector)
    distance_vector = distance_vector / distance_vector.max(axis=0)
    print distance_vector
    return distance_vector
