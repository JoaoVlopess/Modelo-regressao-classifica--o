import numpy as np

class MQOClassifier:
    def init(self):
        self.beta = None
        self.classes = None

    def _one_hot(self, y):
        y = y.flatten()
        self.classes = np.unique(y)
        N = len(y)
        C = len(self.classes)

        Y = np.zeros((N, C))

        for i, c in enumerate(self.classes):
            Y[y == c, i] = 1

        return Y

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y).flatten()

        Y = self._one_hot(y)

        N = X.shape[0]
        X_aug = np.hstack((np.ones((N, 1)), X))

        # Usando pseudoinversa para maior estabilidade numérica
        self.beta = np.linalg.pinv(X_aug) @ Y

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        N = X_test.shape[0]
        X_test_aug = np.hstack((np.ones((N, 1)), X_test))

        scores = X_test_aug @ self.beta
        idx = np.argmax(scores, axis=1)

        return self.classes[idx]



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
        self.det_sigma = None
        self.inv_sigma = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        N, p = X.shape
        self.means = [] 
        self.priors = [] 
        
        for c in self.classes:
            X_j = X[y.flatten() == c]
            
            self.means.append(np.mean(X_j, axis=0))
            
            n_j = X_j.shape[0]

            self.priors.append(n_j / N)

        self.shared_cov = np.cov(X, rowvar=False)
        
        self.shared_cov += np.eye(p) * 1e-6

        self.det_sigma = np.linalg.det(self.shared_cov)
        self.inv_sigma = np.linalg.inv(self.shared_cov)

    def _pdf(self, x, mu):

        p = len(mu)
        
        diff = x - mu
        exponent = -0.5 * (diff @ self.inv_sigma @ diff.T)
        norm = 1 / (np.sqrt((2 * np.pi)**p * self.det_sigma))
        
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
    


class GaussianClassifierPooledCovarianceMatrix:
    def __init__(self):
        self.means = []
        self.matrix_cov = None 
        self.priors = []
        self.classes = None
        self.inv_sigma = None 
        self.det_sigma = None 

    def fit(self, X, y):
        self.classes = np.unique(y)
        N, p = X.shape
        self.means = []  
        self.priors = [] 
        all_covs = []

        for c in self.classes:
            X_j = X[y.flatten() == c]
            
            self.means.append(np.mean(X_j, axis=0))
            
            n_j = X_j.shape[0]
            c_cov = np.cov(X_j, rowvar=False)
            all_covs.append(c_cov * (n_j - 1))
            
            self.priors.append(n_j / N)

        self.matrix_cov = sum(all_covs) / (N - len(self.classes))
        
        self.matrix_cov += np.eye(p) * 1e-6

        self.det_sigma = np.linalg.det(self.matrix_cov)
        self.inv_sigma = np.linalg.inv(self.matrix_cov)

    def _pdf(self, x, mu):

        p = len(mu)
        
        diff = x - mu
        exponent = -0.5 * (diff @ self.inv_sigma @ diff.T)
        norm = 1 / (np.sqrt((2 * np.pi)**p * self.det_sigma))
        
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
    


class GaussianClassifierFriedman:
    def __init__(self):
        self.means = []
        self.priors = []
        self.covs = []
        self.matrix_cov = None 
        self.classes = None

    def fit(self, X, y, lamb):
        self.classes = np.unique(y) 
        N, p = X.shape
        all_covs = []
        individual_covs = []
        temp_n_j = []

        for c in self.classes:
            X_j = X[y.flatten() == c]
            n_j = X_j.shape[0]
            temp_n_j.append(n_j)
            
            self.means.append(np.mean(X_j, axis=0))
            self.priors.append(X_j.shape[0] / N)

            c_cov = np.cov(X_j, rowvar=False)
            individual_covs.append(c_cov)

            all_covs.append(c_cov * (n_j - 1))
        
        self.matrix_cov = sum(all_covs) / (N - len(self.classes))
        self.matrix_cov += np.eye(p) * 1e-6

        friedman_covs = []
        for i, c in enumerate(self.classes):
            n_j = temp_n_j[i]
            ind_cov = individual_covs[i]

            friedman_covs.append(((1-lamb) * (n_j * ind_cov) + (N * lamb *  self.matrix_cov)) / ((1-lamb) * n_j + (lamb * N) ))

        self.covs = friedman_covs

    def _pdf(self, x, mu, sigma):
            p = len(mu)
            det_sigma = np.linalg.det(sigma)
            inv_sigma = np.linalg.inv(sigma)
            
            diff = x - mu
            exponent = -0.5 * (diff @ inv_sigma @ diff.T)
            norm = 1 / (np.sqrt((2 * np.pi)**p * det_sigma))
            
            return norm * np.exp(exponent)
    
    def predict(self, X_test):
            predictions = []
            for x in X_test:
                posteriors = []
                for i in range(len(self.classes)):
                    prob = self._pdf(x, self.means[i], self.covs[i]) * self.priors[i]
                    posteriors.append(prob)
                
                predictions.append(self.classes[np.argmax(posteriors)])
            return np.array(predictions)
    

class  BayesClassifier:
    def __init__(self):
        self.means = []
        self.priors = []
        self.covs = []
        self.classes = None

    def fit(self, X, y,):
        self.classes = np.unique(y) 
        N, p = X.shape

        for c in self.classes:
            X_j = X[y.flatten() == c]
            
            self.means.append(np.mean(X_j, axis=0))
            self.priors.append(X_j.shape[0] / N)

            variancias = np.var(X_j, axis=0)
            matrix_diagonal = np.diag(variancias)

            matrix_diagonal += np.eye(p) * 1e-6

            self.covs.append(matrix_diagonal)

    def _pdf(self, x, mu, sigma):
        p = len(mu)

        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma)

        diff = x - mu

        exponent = -0.5 * (diff @ inv_sigma @ diff.T)

        norm = 1 / (np.sqrt((2 * np.pi)**p * det_sigma))

        return norm * np.exp(exponent)
    
    def predict(self, X_test):
        predictions = []
        
        
        for x in X_test:
            posteriors = []
            
            for i in range(len(self.classes)):
                
                prob = self._pdf(x, self.means[i], self.covs[i]) * self.priors[i]
                posteriors.append(prob)
            
            
            idx_max = np.argmax(posteriors)
            predictions.append(self.classes[idx_max])
            
        return np.array(predictions)




    
