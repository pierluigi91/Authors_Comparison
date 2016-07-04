import clean_text
from pyspark import SparkConf, SparkContext
import re
import json


file = open("/home/pierluigi/Scrivania/testi/Charles_Darwin/cleaned_text_cleaned_text_cleaned_text_On the Origin of Species By Means of Natural Selection.txt", "r")
out = open("/home/pierluigi/Scrivania/testi/prove/OUT.txt", "w")
r=""
for line in file:
    r+=line


r2 = ' '.join(unicode(r, 'utf-8').splitlines())
file2 = clean_text.split_into_sentences(r2)
for line in file2:
    out.write(re.sub("[^a-zA-Z\s]"," ",clean_text.stemming(line).strip()))
    out.write("\n")




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


map_red_count_x_occ("/home/pierluigi/Scrivania/testi/prove/OUT.txt",1)