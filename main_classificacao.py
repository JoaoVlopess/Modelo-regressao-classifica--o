from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from Classification import GaussianClassifier, GaussianClassifierSharedCov
from LinearRegression import MeanModel
from LinearRegression import LinearRegression
from LinearRegression import TrainTest

# Dados com problemas de load no quesito shape (Tive que colocar a transposta)
data = np.loadtxt("./data/EMGsDataset (2).csv",delimiter=',').T 

N = data.shape[0]
C = 5
y_original = data[:,-1:]
z = y_original.flatten()
classes = [1,2,3,4,5]


classes_nomes = [
    'Neutro',
    'Sorriso',
    'Sobrancelhas levantadas',
    'Surpreso',
    'Rabugento',
]

cores = [
    'red',
    'blue',
    'green',
    'orange',
    'yellow'
]


# Var modelo MQO
Y_M = np.zeros((N,5))
indices_colunas = y_original.astype(int).flatten() - 1
Y_M[np.arange(N), indices_colunas] = 1
X_M = np.array(data[:,:-1])

# Var modelo Gaussiano
Y_G = Y_M.T
X_G = X_M.T

fig = plt.figure(figsize=(10, 7))

x1 = X_M[:,0]
x2 = X_M[:,1]

for i, classe in enumerate(classes):
    indices = (z == classe)
    plt.scatter(x1[indices], x2[indices], c=cores[i],label=classes_nomes[i],edgecolors='k', cmap='viridis', marker='o')

plt.title('Visualização 2D dos Dados de EMG') 

clf = GaussianClassifier()
clf.fit(X_M, z)

x_min, x_max = X_M[:, 0].min() - 100, X_M[:, 0].max() + 100
y_min, y_max = X_M[:, 1].min() - 100, X_M[:, 1].max() + 100

xx, yy = np.meshgrid(np.arange(x_min, x_max, 50),
                     np.arange(y_min, y_max, 50))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z_pred = clf.predict(grid_points)
Z_pred = Z_pred.reshape(xx.shape)

plt.figure(figsize=(10, 7))

plt.contourf(xx, yy, Z_pred, alpha=0.3, cmap='rainbow')

for i, classe in enumerate(classes):
    indices = (z == classe)
    plt.scatter(x1[indices], x2[indices], c=cores[i], label=classes_nomes[i], edgecolors='k')

plt.title('Fronteiras de Decisão - Classificador Gaussiano tradicional')
plt.xlabel('Sensor 1')
plt.ylabel('Sensor 2')
plt.legend()

clf_linear = GaussianClassifierSharedCov()
clf_linear.fit(X_M, z)


x_min, x_max = X_M[:, 0].min() - 100, X_M[:, 0].max() + 100
y_min, y_max = X_M[:, 1].min() - 100, X_M[:, 1].max() + 100

xx, yy = np.meshgrid(np.arange(x_min, x_max, 50),
                     np.arange(y_min, y_max, 50))

grid_points = np.c_[xx.ravel(), yy.ravel()]

Z_pred = clf_linear.predict(grid_points)
Z_pred = Z_pred.reshape(xx.shape)

plt.figure(figsize=(12, 8))

plt.contourf(xx, yy, Z_pred, alpha=0.3, cmap='rainbow' )

for i, classe in enumerate(classes):
    indices = (z == classe)
    plt.scatter(x1[indices], x2[indices], 
                c=cores[i], 
                label=classes_nomes[i], 
                edgecolors='k', 
                s=20) 

plt.title('Fronteiras de Decisão Lineares (Covariâncias Iguais)')
plt.xlabel('Sensor 1')
plt.ylabel('Sensor 2')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()