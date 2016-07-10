from threading import Thread
from machine_learning.naive_bayes.training import train as nb_tr
from machine_learning.spark import spark_text as sp



def train():
    Thread(target=nb_tr.train())
    #lancio training di spark
    Thread(target=sp.train_vectors())
    #dp_tr.training()


train()
