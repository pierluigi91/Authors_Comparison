from threading import Thread
from deep_learning import train as dp_tr
from machine_learning.naive_bayes.training import train as nb_tr
from machine_learning.spark import spark_text as sp



def train():
    #Thread(target=dp_tr.training())
    Thread(target=nb_tr.train())
    #lancio training di spark
    Thread(target=sp.train_vectors())



train()
