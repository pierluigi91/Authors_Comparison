from __future__ import division
import operator
from functools import reduce

from ExceptionNotSeen import NotSeen

class Classifier(object):
    def __init__(self, trainedData, tokenizer):
        super(Classifier, self).__init__()
        self.data = trainedData
        self.tokenizer = tokenizer
        self.defaultProb = 0.000000001

    def  classify(self, text):
        
        documentCount = self.data.getDocCount()
        classes = self.data.getClasses()
        tokens = self.tokenizer.tokenize(text)
        probsOfClasses = {}

        for className in classes:
            #qui viene calcolata la probablita' che un termine occorra nel prossimo testo di questa classe
            # P(Token_1|Class_i)
            tokensProbs = [self.getTokenProb(token, className) for token in tokens]
            #qui viene calcolata la probabilita' che un set di token occorra nel prossimo testo di questa classe con
            #una piccola modifica della formula standard volta alla normalizzazione
            # P(Token_1|Class_i) * P(Token_2|Class_i) * ... * P(Token_n|Class_i)
            try:
                tokenSetProb = reduce(lambda a,b: (a*b)/(a+b), (i for i in tokensProbs if i) )
            except:
                tokenSetProb = 0
            probsOfClasses[className] = tokenSetProb / self.getPrior(className)
        return sorted(probsOfClasses.items(),
            key=operator.itemgetter(1),
            reverse=True)


    def getPrior(self, className):
        return self.data.getClassDocCount(className) /  self.data.getDocCount()

    def getTokenProb(self, token, className):
        #p(token|Class_i)
        classDocumentCount = self.data.getClassDocCount(className)
        #se il token non e' presente nel training set, allora non sara' neppure indicizzato
        #quindi e' ritornato il valore None e non viene incluso nei calcoli
        try:
            tokenFrequency = self.data.getFrequency(token, className)
        except NotSeen as e:
            return None
        #questo significa che il token non occorre in questa classe ma in altre
        if tokenFrequency is None:
            return self.defaultProb

        probablity =  tokenFrequency / classDocumentCount
        return probablity
