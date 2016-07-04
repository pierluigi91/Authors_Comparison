#!/usr/bin/python -S
# coding=utf-8
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import re
import os
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
from pyspark import SparkConf, SparkContext
import json


rootdir = '/home/pierluigi/Scrivania/testi'

index=0
caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def clean_top_bottom(path, subdir):
    start = False
    text = open(path, "r")
    libro = text.readlines()
    # os.path.basename(path)
    cleaned_text = open(subdir+"/cleaned_text_"+os.path.basename(path).replace(".txt","")+".txt","w")
    for i in range(0, len(libro)):
        if libro[i].__contains__("*** START OF") or libro[i].__contains__("***START OF"):
            start=True
        if libro[i].__contains__("*** END OF THIS PROJECT GUTENBERG ") or libro[i].__contains__("*** END OF THE PROJECT GUTENBERG ") \
                              or libro[i].__contains__("***END OF THIS PROJECT GUTENBERG ") or libro[i].__contains__("***END OF THE PROJECT GUTENBERG "):
            start=False
        if start:
            cleaned_text.write(libro[i])
    cleaned_text.close()
    text.close()





def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = text.replace("...","<prd><prd>.")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def smart_truncate(content, length=100, suffix='...'):
    if len(content) <= length:
        return content
    else:
        return ' '.join(content[:length+1].split(' ')[0:-1]) + suffix


def stopping(linea):
    stop = stopwords.words('english')
    result = ""
    for i in linea.split():
        if i not in stop:
            result += i + " "
    return result.strip()

def stemming(linea):
    stemmer = SnowballStemmer("english")
    result=""
    for word in linea.split(' '):
        result += stemmer.stem(word) + " "
    return result.strip()



def parsing():
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.__contains__("cleaned"):
                input = open(subdir+"/"+file, 'r')
                text = input.read()
                text = ' '.join(unicode(text, 'utf-8').splitlines())
                print file
                #out2 = open('ul2.txt', 'w')
                #out2.write(text)
                output = split_into_sentences(text.encode('utf-8'))
                #output2 = [[stemmer.stem(word) for word in sentence.split(' ')] for sentence in output]
                output2= [[word for word in sentence.split(' ')] for sentence in output]
                print(len(output2))
                out = open(subdir+"/"+file.replace(".txt", "")+"_modified.txt", 'w')
                output3 = []
                for i in output2:
                    linea = ''
                    for j in i:
                        linea += j + ' '
                    #linea += '\n'
                    output3.append(linea)
                for i in output3:
                    if len(i) >= 12:
                        temp = re.sub("[^a-zA-Z0-9\s]"," ",i)
                        temp = stemming(temp)
                        #map_red(temp)
                        temp = stopping(temp)
                        out.write(smart_truncate(str(temp), length=256, suffix=' ')+"\n")


# for i in os.listdir("/home/pierluigi/Scrivania/testi"):
#     if i.endswith(".txt"):
#         clean_top_bottom("/home/pierluigi/Scrivania/testi/"+i)
# parsing()



# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         clean_top_bottom(os.path.abspath(subdir)+"/"+file, subdir)
#         print os.path.join(subdir, file)
# parsing()

