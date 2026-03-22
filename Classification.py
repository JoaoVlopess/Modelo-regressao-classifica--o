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
        
        
        for x in X_test:
            posteriors = []
            
            for i in range(len(self.classes)):
                
                prob = self._pdf(x, self.means[i], self.covs[i]) * self.priors[i]
                posteriors.append(prob)
            
            
            idx_max = np.argmax(posteriors)
            predictions.append(self.classes[idx_max])
            
        return np.array(predictions)

class GaussianClassifierSharedCov:
    def __init__(self):
        self.means = []
        self.shared_cov = None 
        self.priors = []
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        N, p = X.shape
        all_covs = []
        
        for c in self.classes:
            X_j = X[y.flatten() == c]
            
            self.means.append(np.mean(X_j, axis=0))
            
            n_j = X_j.shape[0]
            c_cov = np.cov(X_j, rowvar=False)
            all_covs.append(c_cov * (n_j - 1))
            
            
            self.priors.append(n_j / N)

        self.shared_cov = sum(all_covs) / (N - len(self.classes))
        
        self.shared_cov += np.eye(p) * 1e-6

    def _pdf(self, x, mu):

        p = len(mu)
        det_sigma = np.linalg.det(self.shared_cov)
        inv_sigma = np.linalg.inv(self.shared_cov)
        
        diff = x - mu
        exponent = -0.5 * (diff @ inv_sigma @ diff.T)
        norm = 1 / (np.sqrt((2 * np.pi)**p * det_sigma))
        
        return norm * np.exp(exponent)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            posteriors = []
            for i in range(len(self.classes)):

                prob = self._pdf(x, self.means[i]) * self.priors[i]
                posteriors.append(prob)
            
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)