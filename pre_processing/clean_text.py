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
import time


sentence2line_outputdir = '../data/sentence_to_line_data/'
stemmed_outputdir = '../data/stemmed_data/'
index=0
caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def clean_top_bottom(path):
    start = False
    text = open(path, "r")
    libro = text.readlines()
    # os.path.basename(path)
    # cleaned_text = open(subdir+"/cleaned_text_"+os.path.basename(path).replace(".txt","")+".txt","w")
    cleaned_text = ''
    for i in range(0, len(libro)):
        if libro[i].__contains__("*** START OF") \
                or libro[i].__contains__("***START OF"):
            start = True
        if libro[i].__contains__("*** END OF THIS PROJECT GUTENBERG ") \
                or libro[i].__contains__("*** END OF THE PROJECT GUTENBERG ") \
                or libro[i].__contains__("***END OF THIS PROJECT GUTENBERG ") \
                or libro[i].__contains__("***END OF THE PROJECT GUTENBERG "):
            start = False
        if start:
            cleaned_text += libro[i]
            # cleaned_text.write(libro[i])
    #cleaned_text.close()
    text.close()
    return cleaned_text



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


def parsing(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.__contains__("cleaned"):
                input = open(subdir+"/"+file, 'r')
                text = input.read()
                text = ' '.join(unicode(text, 'utf-8').splitlines())
                print file
                # out2 = open('ul2.txt', 'w')
                # out2.write(text)
                output = split_into_sentences(text.encode('utf-8'))
                # output2 = [[stemmer.stem(word) for word in sentence.split(' ')] for sentence in output]
                output2 = [[word for word in sentence.split(' ')] for sentence in output]
                print(len(output2))
                out = open(subdir+"/"+file.replace(".txt", "")+"_modified.txt", 'w')
                output3 = []
                for i in output2:
                    linea = ''
                    for j in i:
                        linea += j + ' '
                    # linea += '\n'
                    output3.append(linea)
                for i in output3:
                    if len(i) >= 12:
                        temp = re.sub("[^a-zA-Z0-9\s]", " ", i)
                        temp = stemming(temp)
                        # map_red(temp)
                        temp = stopping(temp)
                        out.write(smart_truncate(str(temp), length=256, suffix=' ')+"\n")


def remove_empty_lines(text):
    cleaned_text = os.linesep.join([s for s in text.splitlines() if s])
    return cleaned_text


def merge_files_by_author(root_path):
    for subdir, dirs, files in os.walk(root_path):
        author = subdir.split('/')[-1].lower()
        if author != root_path.split('/')[-1]:

            with open(root_path + '/' + author + '.txt', 'w') as output:
                for _file in files:
                    with open(subdir + '/' + _file) as infile:
                        for line in infile:
                            output.write(line)




def process_dataset(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        # cleaned_text = ''
        author = subdir.split('/')[-1]
        for _file in files:
            ts = time.time()
            path_to_file = subdir + '/' + _file
            cleaned_text = clean_top_bottom(path_to_file)
            cleaned_text = ' '.join(unicode(cleaned_text, 'utf-8').splitlines())
            sentence2line_list = split_into_sentences(cleaned_text.encode('utf-8'))
            if not os.path.exists(sentence2line_outputdir):
                os.makedirs(sentence2line_outputdir)
            if not os.path.exists(sentence2line_outputdir + author):
                os.makedirs(sentence2line_outputdir + author)
            # sentence2line_text = [[word for word in sentence.split(' ')] for sentence in sentence2line_list]
            output_splitline_file = open(sentence2line_outputdir + author + '/' + str(_file), 'w')
            output_splitline_text = ''
            for line in sentence2line_list:
                output_splitline_text += smart_truncate(re.sub('\s+', ' ', line), length=256, suffix='') + '\n'
            output_splitline_text = remove_empty_lines(output_splitline_text)
            output_splitline_file.write(output_splitline_text)
            output_splitline_file.close()

            if not os.path.exists(stemmed_outputdir):
                os.makedirs(stemmed_outputdir)
            if not os.path.exists(stemmed_outputdir + author):
                os.makedirs(stemmed_outputdir + author)
            output_stemmed_file = open(stemmed_outputdir + author + '/' + str(_file), 'w')
            output_stemmed_text = ''
            for line in output_splitline_text.splitlines():
                temp = stopping(line)
                temp = stemming(temp)
                temp = re.sub('[^a-zA-Z\s]', '', temp)
                temp = smart_truncate(temp, length=256, suffix='') + '\n'
                output_stemmed_text += temp
            output_stemmed_text = remove_empty_lines(output_stemmed_text)
            output_stemmed_file.write(output_stemmed_text)
            output_stemmed_file.close()
            print author.replace('_', ' ') + ', ' + _file + ' in ' + str(time.time() - ts) + ' secondi.'

#process_dataset('../data/dataset')
merge_files_by_author('../data/sentence_to_line_data')






# for i in os.listdir("/home/pierluigi/Scrivania/testi"):
#     if i.endswith(".txt"):
#         clean_top_bottom("/home/pierluigi/Scrivania/testi/"+i)
# parsing()



# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         clean_top_bottom(os.path.abspath(subdir)+"/"+file, subdir)
#         print os.path.join(subdir, file)
# parsing()

