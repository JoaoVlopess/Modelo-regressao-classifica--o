import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import MeanModel

# REGRESSÃO parte 1/2
data = np.loadtxt('data/aerogerador (1).dat')
X = data[:,:-1]
y = data[:,-1:]

plt.scatter(X, y, edgecolors = '0')

#REGRESSÃO parte 3

modelo_media = MeanModel()
modelo_media.fit(X, y)

y_hat = modelo_media.predict([[14], [10], [3], [5]])
plt.scatter(X, y, color='blue', label='Dados Reais', edgecolors='0')
plt.plot([[14], [10], [3], [5]], y_hat, color='red', linewidth=3, label='Modelo de Médias')
plt.legend()
plt.title("Visualização do Modelo de Médias")
plt.show()