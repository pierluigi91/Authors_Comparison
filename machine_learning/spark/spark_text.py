from pyspark import SparkConf, SparkContext
from nltk.tag import pos_tag
from nltk import word_tokenize
import time

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


def pos_tagging(path, split_size, sc): # DA FARE SUL TESTO PRE-STEMMING
    ts = time.time()

    text = sc.textFile(path, 128)
    num_sentences = text.count()
    #FIRST ATTEMPT
    # mapred_01 = text.map(lambda line: pos_tag(word_tokenize(line))).reduce(lambda x,y:x+y)
    # mapred_02 = sc.parallelize(mapred_01).map(lambda (a,b):(b,a)).reduceByKey(lambda x, y: (str(x)+" "+str(y)))
    # #mapred_02.saveAsTextFile("TAGGED")
    # mapred_03 = mapred_02.map(lambda (a,b): (a, len(list(b.split()))))
    # #mapred_03.saveAsTextFile("COUNT")
    # num_conj = mapred_03.collectAsMap().get('IN')
    # print num_conj/float(num_sentences)

    #SECOND ATTEMPT
    # mapred = text.flatMap(lambda line: [tag for word, tag in pos_tag(word_tokenize(line))]).filter(lambda line: 'IN' in line).count()
    # print mapred

    #PORCODIO ATTEMPT
    mapred = text.flatMap(lambda line: pos_tag(line.split())).filter(lambda line: 'IN' in line)
    print 'RISULTATO: ' + str(mapred.count())
    print 'SPLIT: ' + str(split_size)
    print 'TEMPO: ' + str(time.time()-ts)

splits = [1, 2, 4, 8, 16, 32, 64, 128]

conf = (SparkConf().setMaster('local[*]').setAppName('mapred'))
sc = SparkContext(conf=conf)

for s in splits:
    pos_tagging("../../data/input_stemmed/arthur_conan_doyle.txt", s, sc)
#pos_tagging("/home/pierluigi/Scaricati/sub_conj.txt")


