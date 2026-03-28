import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Classification import (BayesClassifier, GaussianClassifier, GaussianClassifierFriedman, 
                            GaussianClassifierPooledCovarianceMatrix, 
                            GaussianClassifierSharedCov, MQOClassifier)


data = np.loadtxt("./data/EMGsDataset (2).csv", delimiter=',').T 
qnt_amostras = data.shape[0]
X_M = data[:, :-1]
y_original = data[:, -1:]
z = y_original.flatten()

classes = [1, 2, 3, 4, 5]
classes_nomes = [
    'Neutro', 
    'Sorriso', 
    'Sobrancelhas levantadas', 
    'Surpreso', 
    'Rabugento'
    ]

cores = [
    'red', 
    'blue', 
    'green', 
    'orange', 
    'yellow'
    ]

meu_cmap = ListedColormap(cores)


def plot_decision_boundaries(model, X, y, title):
    """Gera o gráfico de fronteiras para qualquer modelo passado."""
    x_min, x_max = X[:, 0].min() - 100, X[:, 0].max() + 100
    y_min, y_max = X[:, 1].min() - 100, X[:, 1].max() + 100
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 50),
                         np.arange(y_min, y_max, 50))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_pred = model.predict(grid_points)
    Z_pred = Z_pred.reshape(xx.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z_pred, alpha=0.2, cmap=meu_cmap)
    
    for i, classe in enumerate(classes):
        idx = (y == classe)
        plt.scatter(X[idx, 0], X[idx, 1], c=cores[i], label=classes_nomes[i], 
                    edgecolors='k', s=25, alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Sensor 1')
    plt.ylabel('Sensor 2')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)

meu_lambda = 0.5

modelos = [
    (MQOClassifier(), 'Classificador MQO'),
    (GaussianClassifier(), 'Classificador Gaussiano Tradicional'),
    (GaussianClassifierSharedCov(), 'Classificador Gaussiano com Covariâncias Iguais'),
    (GaussianClassifierPooledCovarianceMatrix(), 'Classificador Gaussiano com Matriz de Covariância Agregada'), 
    (GaussianClassifierFriedman(), 'Classificador Gaussiano Friedman'),
    (BayesClassifier(), 'Classificador de Bayes Ingênuo')
]

sensor1 = X_M[:, 0]
sensor2 = X_M[:, 1]

for i, classe in enumerate(classes):

    indices = (z == classe)
    
    plt.scatter(sensor1[indices],
                sensor2[indices],
                c=cores[i],
                label=classes_nomes[i],
                edgecolors='k',
                alpha=0.6,       
                s=20)           

plt.title('Distribuição dos Sinais de EMG por Classe')
plt.xlabel('Sensor 1')
plt.ylabel('Sensor 2')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()


for clf, nome in modelos:
    if isinstance(clf, MQOClassifier):
        clf.fit(X_M, z)
    else:
        if isinstance(clf, GaussianClassifierFriedman):
            clf.fit(X_M.T, z, lamb=meu_lambda)
        else:
            clf.fit(X_M.T, z)
            
    plot_decision_boundaries(clf, X_M, z, nome)

plt.show()


def calculate_accuracy(real, predicts):
    return np.mean(real == predicts)


def validation_monte_carlo(model ,X , y_z, rodadas=500):
    qnt_amostras = data.shape[0]
    corte = int(qnt_amostras*0.8)
    accuracy_list = []

    for i in range(rodadas):
        indexes = np.random.permutation(qnt_amostras)

        idx_train = indexes[:corte]
        idx_test = indexes[corte:]

        X_train, y_train = X[idx_train], y_z[idx_train]
        X_test, y_test = X[idx_test], y_z[idx_test]

        if isinstance(model, GaussianClassifierFriedman):
            model.fit(X_train, y_train, lamb=meu_lambda)
        else:
            model.fit(X_train, y_train)

        predicts = model.predict(X_test)

        accuracy = calculate_accuracy(y_test, predicts)
        accuracy_list.append(accuracy)

    return np.mean(accuracy_list), np.std(accuracy_list)

for clf, nome in modelos:
    mean, dp = validation_monte_carlo(clf, X_M, z)
    
    print(f"Modelo: {nome:45} | Acurácia: {mean*100:6.2f}% (+/- {dp*100:.2f}%)")


