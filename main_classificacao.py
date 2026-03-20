import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import MeanModel
from LinearRegression import LinearRegression
from LinearRegression import TrainTest

# Dados com problemas de load no quesito shape (Tive que colocar a transposta)
data = np.loadtxt("./data/EMGsDataset (2).csv",delimiter=',').T 

N = data.shape[0]
C = 5
y_original = data[:,-1:]

# Var modelo MQO
Y_M = np.zeros((N,5))
indices_colunas = y_original.astype(int).flatten() - 1
Y_M[np.arange(N), indices_colunas] = 1
X_M = np.array(data[:,:-1])

# Var modelo Gaussiano
Y_G = Y_M.T
X_G = X_M.T

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x1 = X_M[:,:-1]
x2 = X_M[:,-1:]
z = y_original[:].flatten()

img = ax.scatter(x1, x2, z, c=z, cmap='viridis', marker='o')
ax.set_xlabel('Sensor 1 (X1)')
ax.set_ylabel('Sensor 2 (X2)')
ax.set_zlabel('Classe (Y)')
fig.colorbar(img, ax=ax, label='Classes')

plt.title('Visualização 3D dos Dados de EMG')
plt.show()
 

