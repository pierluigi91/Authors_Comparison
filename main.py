#from deep_learning import predict_last as dp #----> suddividere in metodi di train e eval
from threading import Thread
from machine_learning.naive_bayes.training import train as nb_tr
from machine_learning.naive_bayes.eval import evaluation as nb_ev
from machine_learning.spark import spark_text as sp #suddividere in metodi di train e eval


def train():
    #Thread(target=sp.train_vectors())
    Thread(target=nb_tr.train())


def evaluate(path):  # params
    Thread(target=sp.evaluate(path))
    Thread(target=nb_ev.eval(path))
    #Thread(target=dp_eval.evaluate(path))

train()
#evaluate()

