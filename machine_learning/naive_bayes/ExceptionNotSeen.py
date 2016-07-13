class NotSeen(Exception):
    """
    Eccezione per i token che non sono indicizzati
    perche' non sono occorsi nel training data
    """
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return "Token '{}' is never seen in the training set.".format(self.value)