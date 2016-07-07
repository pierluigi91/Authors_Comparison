#from deep_learning import predict_last as dp #----> suddividere in metodi di train e eval
from threading import Thread
from machine_learning.naive_bayes.training import train as nb_tr
from machine_learning.naive_bayes.eval import evaluation as nb_ev
from machine_learning.spark import spark_text as sp #suddividere in metodi di train e eval


def train():
    #Thread(target=sp)
    Thread(target=nb_tr.train())
    #Thread(target=dp)

def evaluate(): #params
    path = raw_input("Inserire un path di un file da classificare: ")
    Thread(target=sp.average_words_length(path))
    Thread(target=nb_ev.eval(path))
    #Thread(target=tf) tf da implementare

train()
evaluate()