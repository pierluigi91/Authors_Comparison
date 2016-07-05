from pyspark import SparkConf, SparkContext
import re
import numpy as np
import json
from operator import add


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
    return float(mapred_02[1])/float(unique_words), float(mapred_02[2])/float(unique_words)


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


def get_spark_vector(path):
    conf = (SparkConf().setMaster('local').setAppName('mapred'))
    sc = SparkContext(conf=conf)
    vector = list()
    vector.append(hapax_dix_legomena(sc, path)[0])
    vector.append(hapax_dix_legomena(sc, path)[1])
    vector.append(average_words_length(sc, path))
    vector.append(average_sentences_length(sc, path))
    print vector
    return np.array(vector)

get_spark_vector("/Users/Max/Desktop/prova.txt")
