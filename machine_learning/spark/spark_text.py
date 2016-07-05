from pyspark import SparkConf, SparkContext
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

def pos_tagging(path): # DA FARE SUL TESTO PRE-STEMMING
    conf = (SparkConf().setMaster('local').setAppName('mapred'))
    sc = SparkContext(conf=conf)
    text = sc.textFile(path,128)
    num_sentences = text.count()
    mapred_01 = text.map(lambda line: pos_tag(word_tokenize(line))).reduce(lambda x,y:x+y)
    mapred_02 = sc.parallelize(mapred_01).map(lambda (a,b):(b,a)).reduceByKey(lambda x, y: (str(x)+" "+str(y)))
    #mapred_02.saveAsTextFile("TAGGED")
    mapred_03 = mapred_02.map(lambda (a,b): (a, len(list(b.split()))))
    #mapred_03.saveAsTextFile("COUNT")
    num_conj = mapred_03.collectAsMap().get('IN')
    print num_conj/float(num_sentences)

pos_tagging("../../data/input_stemmed/arthur_conan_doyle.txt")
#pos_tagging("/home/pierluigi/Scaricati/sub_conj.txt")


