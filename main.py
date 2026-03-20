import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import MeanModel
from LinearRegression import LinearRegression
from LinearRegression import TrainTest



# REGRESSÃO parte 1/2
data = np.loadtxt('C:\\Users\\Samsung\\OneDrive\\Área de Trabalho\\Programação\\IA\\Trabalho\\Modelo-regressao-classifica--o\\data\\aerogerador (1).dat')
X = data[:,:-1]
y = data[:,-1:]
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)


#REGRESSÃO parte 3/4
# Criando o modelo de médias

fig1 = plt.figure(1)
modelo_media = MeanModel(y)
modelo_media.fit()

media_y = modelo_media.beta_0

x_reta = np.linspace(0, 14, 100).reshape(-1, 1)

X_teste = [[9], [10], [1], [5]]
y_hat = modelo_media.predict(X_teste)
plt.scatter(X, y, color='blue', label='Dados Reais', edgecolors='0')
plt.axhline(y=modelo_media.beta_0, color='magenta', linestyle='--', label='Média')
plt.plot(X_teste, y_hat, color='cyan', linewidth=3, label='Modelo de Médias')
plt.legend()
plt.title("Visualização do Modelo de Médias")

# MQO TRADICIONAL

fig2 = plt.figure(2)
modelo_mqo = LinearRegression(X, y, intercepto=False, lambda_=0.0) 
modelo_mqo.fit()

y_hat = modelo_mqo.predict(x_line)
plt.scatter(X, y, color='blue', label='Dados Reais', edgecolors='0')
plt.plot(x_line, y_hat, color='red', linewidth=3, label='MQO Tradicional')
plt.legend()
plt.title("Visualização do Modelo MQO Tradicional")


# MQO REGULARIZADO (tikhonov)
fig3 = plt.figure(3)

modelo_tikhonov1 = LinearRegression(X, y, intercepto=False, lambda_=0.25)
modelo_tikhonov2 = LinearRegression(X, y, intercepto=False, lambda_=0.50)
modelo_tikhonov3 = LinearRegression(X, y, intercepto=False, lambda_=0.75)
modelo_tikhonov4 = LinearRegression(X, y, intercepto=False, lambda_= 1.0)

modelo_tikhonov1.fit()
modelo_tikhonov2.fit()
modelo_tikhonov3.fit()
modelo_tikhonov4.fit()

y_hat1 = modelo_tikhonov1.predict(x_line)
y_hat2 = modelo_tikhonov2.predict(x_line)
y_hat3 = modelo_tikhonov3.predict(x_line)
y_hat4 = modelo_tikhonov4.predict(x_line)

plt.scatter(X, y, color='blue', label='Dados Reais', edgecolors='0')
plt.plot(x_line, y_hat1, color='red', linewidth=3, label='MQO Regularizado (Tikhonov)')
plt.plot(x_line, y_hat2, color='blue', linewidth=3, label='MQO Regularizado (Tikhonov)')
plt.plot(x_line, y_hat3, color='green', linewidth=3, label='MQO Regularizado (Tikhonov)')
plt.plot(x_line, y_hat4, color='orange', linewidth=3, label='MQO Regularizado (Tikhonov)')
plt.legend()
plt.title("Visualização do Modelo MQO Regularizado (Tikhonov)")


plt.show()

# REGRESSÃO parte 5
TT = TrainTest()
R = 500

resultados = {
    'MeanModel': {'MSE': [], 'R2': []},
    'MQO': {'MSE': [], 'R2': []},
    'Tikhonov_0.25': {'MSE': [], 'R2': []},
    'Tikhonov_0.50': {'MSE': [], 'R2': []},
    'Tikhonov_0.75': {'MSE': [], 'R2': []},
    'Tikhonov_1.00': {'MSE': [], 'R2': []}
}

for _ in range(R):
    X_train, X_test, y_train, y_test = TT.train_test_split(X, y)

    # 1. MeanModel
    modelo_mean = MeanModel(y_train)
    modelo_mean.fit()
    y_pred = modelo_mean.predict(X_test)

    resultados['MeanModel']['MSE'].append(TT.mean_squared_error(y_test, y_pred))
    resultados['MeanModel']['R2'].append(TT.r2_score(y_test, y_pred))

    # 2. MQO tradicional
    modelo_mqo = LinearRegression(X_train, y_train, intercepto=True, lambda_=0.0)
    modelo_mqo.fit()
    y_pred = modelo_mqo.predict(X_test)

    resultados['MQO']['MSE'].append(TT.mean_squared_error(y_test, y_pred))
    resultados['MQO']['R2'].append(TT.r2_score(y_test, y_pred))

    # 3. Tikhonov lambda = 0.25
    modelo_t1 = LinearRegression(X_train, y_train, intercepto=True, lambda_=0.25)
    modelo_t1.fit()
    y_pred = modelo_t1.predict(X_test)

    resultados['Tikhonov_0.25']['MSE'].append(TT.mean_squared_error(y_test, y_pred))
    resultados['Tikhonov_0.25']['R2'].append(TT.r2_score(y_test, y_pred))

    # 4. Tikhonov lambda = 0.50
    modelo_t2 = LinearRegression(X_train, y_train, intercepto=True, lambda_=0.50)
    modelo_t2.fit()
    y_pred = modelo_t2.predict(X_test)

    resultados['Tikhonov_0.50']['MSE'].append(TT.mean_squared_error(y_test, y_pred))
    resultados['Tikhonov_0.50']['R2'].append(TT.r2_score(y_test, y_pred))

    # 5. Tikhonov lambda = 0.75
    modelo_t3 = LinearRegression(X_train, y_train, intercepto=True, lambda_=0.75)
    modelo_t3.fit()
    y_pred = modelo_t3.predict(X_test)

    resultados['Tikhonov_0.75']['MSE'].append(TT.mean_squared_error(y_test, y_pred))
    resultados['Tikhonov_0.75']['R2'].append(TT.r2_score(y_test, y_pred))
    
     # 6. Tikhonov lambda = 1.00
    modelo_t4 = LinearRegression(X_train, y_train, intercepto=True, lambda_=1.00)
    modelo_t4.fit()
    y_pred = modelo_t4.predict(X_test)

    resultados['Tikhonov_1.00']['MSE'].append(TT.mean_squared_error(y_test, y_pred))
    resultados['Tikhonov_1.00']['R2'].append(TT.r2_score(y_test, y_pred))