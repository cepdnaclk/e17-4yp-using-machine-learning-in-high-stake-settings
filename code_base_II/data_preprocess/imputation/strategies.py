
class DataImputer:
    
    def __init__(self, strategy):
        self.strategy = strategy
        
    def impute(self, data):
        return self.strategy.impute(data)
    

