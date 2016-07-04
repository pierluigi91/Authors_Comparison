from pyspark import SparkConf, SparkContext
import re
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

def map_red_count_x_occ(path, x):
    conf = (SparkConf().setMaster('local').setAppName('mapred'))
    sc = SparkContext(conf=conf)
    text = sc.textFile(path)
    mapred_01 = text.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word.encode("utf-8"), 1)) \
        .reduceByKey(lambda a, b: a + b)
    mapred_02 = dict(mapred_01.map(lambda (a, b): (b, a)).countByKey().items())

    print mapred_02[x]

def average_words_length(path):
    conf = (SparkConf().setMaster('local').setAppName('mapred'))
    sc = SparkContext(conf=conf)
    text = sc.textFile(path)
    mapred_01 = text.flatMap(lambda line: line.split(" ")).map(lambda a: (a.encode("utf-8"), (len(a))))
    tot = mapred_01.values().sum()
    print tot/float(mapred_01.count())

    mapred_01.saveAsTextFile("DIOBELLO")


def average_sentences_length(path):
    conf = (SparkConf().setMaster('local').setAppName('mapred'))
    sc = SparkContext(conf=conf)
    text = sc.textFile(path)
    mapred_01 = text.map(lambda line: len(line.split()))
    print mapred_01.mean()

average_sentences_length("../../data/input_backup/arthur_conan_doyle.txt")
