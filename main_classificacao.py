import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Classification import (GaussianClassifier, GaussianClassifierFriedman, 
                            GaussianClassifierPooledCovarianceMatrix, 
                            GaussianClassifierSharedCov)


data = np.loadtxt("./data/EMGsDataset (2).csv", delimiter=',').T 
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
    (GaussianClassifier(), 'Classificador Gaussiano Tradicional'),
    (GaussianClassifierSharedCov(), 'Classificador Gaussiano com Covariâncias Iguais'),
    (GaussianClassifierPooledCovarianceMatrix(), 'Classificador Gaussiano com Matriz de Covariância Agregada'), 
    (GaussianClassifierFriedman(), 'Classificador Gaussiano Friedman')
]

for clf, nome in modelos:
    if isinstance(clf, GaussianClassifierFriedman):
        clf.fit(X_M, z, lamb=meu_lambda)
    else:
        clf.fit(X_M, z)
    plot_decision_boundaries(clf, X_M, z, nome)

plt.show()