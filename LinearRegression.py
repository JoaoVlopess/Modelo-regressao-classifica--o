import numpy as np

# REGRESSÃO parte 3
class MeanModel:
    def __init__(self): 
        self.mean_value = None

    def fit(self,X_train, y_train):
        """
        Essa função é responsável por calcular a média de y e guardar no objeto
        """
        self.mean_value = np.sum(y_train)/np.shape(y_train)[0]
        return self
    
    def predict (self, X_train):
        """
        Essa função é responsável por fazer a predição do modelo
        """
        return np.full(shape=(len(X_train) ,1), fill_value=self.mean_value)

