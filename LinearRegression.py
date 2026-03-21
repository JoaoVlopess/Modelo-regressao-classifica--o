import numpy as np

# REGRESSÃO parte 3


class MeanModel:
    def __init__(self, y_train):
        self.y_train = y_train

    def fit(self):
        """
        Essa função é responsável por calcular a média de y e guardar no objeto
        """
        self.beta_0 = np.mean(self.y_train)

    def predict(self, X_test):
        """
        Essa função é responsável por fazer a predição do modelo
        """
        return np.full(shape=(len(X_test), 1), fill_value=self.beta_0)


class LinearRegression:
    def __init__(self, X_train, y_train, intercepto=True, lambda_=0.0):
        self.X_train = X_train
        self.y_train = y_train
        self.intercepto = intercepto
        self.lambda_ = lambda_
        if self.y_train.ndim == 1:
            self.y_train = self.y_train.reshape(-1, 1)
        self.N, self.p = X_train.shape
        if self.intercepto:
            self.X_train = np.hstack((
                np.ones((self.N, 1)),
                self.X_train
            ))

        self.beta_hat = None

    def fit(self):
        n_features = self.X_train.shape[1]
        I = np.eye(n_features)
        if self.intercepto:
            I[0, 0] = 0
        self.beta_hat = np.linalg.inv(
            self.X_train.T @ self.X_train + self.lambda_ * I) @ self.X_train.T @ self. y_train
        
    def predict(self, X_test):
        X_test = np.array(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        N = X_test.shape[0]
        if self.intercepto:
            if len(X_test.shape) > 2:
                X_test = np.concatenate((np.ones((N, N, 1)), X_test), axis=2)
            else:
                X_test = np.hstack((np.ones((N, 1)), X_test))
        return X_test@self.beta_hat


class TrainTest:
    def train_test_split(self, X, y, train_size=0.8):
        N = X.shape[0]
        indices = np.random.permutation(N)

        n_train = int(N * train_size)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        return X_train, y_train, X_test, y_test

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
