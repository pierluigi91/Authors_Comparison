from machine_learning.naive_bayes import trainedData as TrainedData

class Trainer(object):
    
    """docstring per Trainer"""
    def __init__(self, tokenizer):
        super(Trainer, self).__init__()
        self.tokenizer = tokenizer
        self.data = TrainedData.TrainedData()

    def min_num_token(self, text):
        tokens = self.tokenizer.tokenize(text)
        num_tokens=0
        for t in tokens:
            num_tokens+=1
        return num_tokens

    def train(self, text, className, cap):
        """
        migliora il trained data utilizzando il testo e la classe corrente
        """
        self.data.increaseClass(className)
        
        tokens = self.tokenizer.tokenize(text)
        c=0
        for token in tokens:
            if(c<=cap):
                self.data.increaseToken(token, className)
                c+=1
