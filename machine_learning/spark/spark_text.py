import clean_text
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
    # inp = open(path, "r")
    # sentences = [[sentence for sentence in re.sub("[^a-zA-Z\s]"," ",clean_text.stemming(line).strip()).split(" ")]for line in inp.readlines()]
    # words=[]
    # for sentence in sentences:
    #     for word in sentence:
    #         if len(word) != 0:
    #             words.append(word)
    # average = sum(len(word) for word in words) / len(words)
    #
    #
    # print average
    mapred_01.saveAsTextFile("DIOBELLO")


average_words_length("../data/input_backup/arthur_conan_doyle.txt")
