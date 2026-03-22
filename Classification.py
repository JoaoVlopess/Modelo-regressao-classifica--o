import numpy as np

class GaussianClassifier:
    def __init__(self):
        self.means = []
        self.covs = []
        self.priors = []
        self.classes = None

    def fit(self, X, y):
        """
        Calcula os parâmetros estatísticos (mi, sigma, prior) para cada classe.
        """
        self.classes = np.unique(y) 
        N = X.shape[0]

        for c in self.classes:
            X_j = X[y.flatten() == c]

            self.means.append(np.mean(X_j, axis=0))

            self.covs.append(np.cov(X_j, rowvar=False))

            self.priors.append(X_j.shape[0] / N)

    def _pdf(self, x, mu, sigma):
        """
        Função interna para calcular a Densidade de Probabilidade Gaussiana.
        """
        p = len(mu)
        sigma = sigma + np.eye(p) * 1e-6
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma)
        

        diff = x - mu
        
        exponent = -0.5 * (diff @ inv_sigma @ diff.T)
        
        norm = 1 / (np.sqrt((2 * np.pi)**p * det_sigma))
        
        return norm * np.exp(exponent)
    
    def predict(self, X_test):
        """
        Classifica novos pontos baseando-se na maior probabilidade a posteriori.
        """
        predictions = []
        
        # Itera sobre cada amostra do conjunto de teste
        for x in X_test:
            posteriors = []
            
            for i in range(len(self.classes)):
                # P(x|C) * P(C)
                prob = self._pdf(x, self.means[i], self.covs[i]) * self.priors[i]
                posteriors.append(prob)
            
            # Escolhe a classe com maior probabilidade
            idx_max = np.argmax(posteriors)
            predictions.append(self.classes[idx_max])
            
        return np.array(predictions)
