# ==============================
#  IMPORTS
# ==============================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from Classification import (
    BayesClassifier,
    GaussianClassifier,
    GaussianClassifierFriedman,
    GaussianClassifierPooledCovarianceMatrix,
    GaussianClassifierSharedCov,
    MQOClassifier
)

# ==============================
#  CARREGAMENTO DOS DADOS
# ==============================
data = np.loadtxt(
    "C:\\Users\\Samsung\\OneDrive\\Área de Trabalho\\Programação\\IA\\Vfinal\\Modelo-regressao-classifica--o\\data\\EMGsDataset (2).csv",
    delimiter=','
).T

X_M = data[:, :-1]
z = data[:, -1].flatten()

# ==============================
#  CONFIGURAÇÕES VISUAIS
# ==============================
classes = [1, 2, 3, 4, 5]
classes_nomes = [
    'Neutro',
    'Sorriso',
    'Sobrancelhas levantadas',
    'Surpreso',
    'Rabugento'
]
cores = ['red', 'blue', 'green', 'orange', 'yellow']
meu_cmap = ListedColormap(cores)

# ==============================
#  VISUALIZAÇÃO INICIAL
# ==============================
plt.figure(figsize=(8, 6))
for i, classe in enumerate(classes):
    idx = (z == classe)
    plt.scatter(
        X_M[idx, 0],
        X_M[idx, 1],
        c=cores[i],
        label=classes_nomes[i],
        edgecolors='k',
        s=20,
        alpha=0.7
    )

plt.title('Distribuição dos Sinais de EMG por Classe')
plt.xlabel('Sensor 1')
plt.ylabel('Sensor 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# ==============================
#  LISTA DE MODELOS
# ==============================
modelos = [
    (MQOClassifier, 'MQO tradicional'),
    (GaussianClassifier, 'Classificador Gaussiano Tradicional'),
    (GaussianClassifierSharedCov, 'Classificador Gaussiano (Cov. de todo cj. treino)'),
    (GaussianClassifierPooledCovarianceMatrix, 'Classificador Gaussiano (Cov. Agregada)'),
    (BayesClassifier, 'Classificador de Bayes Ingênuo (Naive Bayes Classifier)'),
    (GaussianClassifierFriedman, 'Classificador Gaussiano Regularizado (Friedman)')
]

# ==============================
#  VISUALIZAÇÃO DAS FRONTEIRAS DE DECISÃO
# ==============================

def plot_decision_boundaries(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 100, X[:, 0].max() + 100
    y_min, y_max = X[:, 1].min() - 100, X[:, 1].max() + 100

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 20),
        np.arange(y_min, y_max, 20)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_pred = model.predict(grid_points)
    Z_pred = Z_pred.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z_pred, alpha=0.2, cmap=meu_cmap)
    plt.grid(True, linestyle='--', alpha=0.3)

    for i, classe in enumerate(classes):
        idx = (y == classe)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=cores[i],
            label=classes_nomes[i],
            edgecolors='k',
            s=25
        )

    plt.title(title)
    plt.xlabel('Sensor 1')
    plt.ylabel('Sensor 2')
    plt.legend()

# ==============================
#  PLOT DOS MODELOS
# ==============================

for model_class, nome in modelos:
    model = model_class()

    if model_class == GaussianClassifierFriedman:
        model.fit(X_M, z, lamb=0.01)  # pode usar valor fixo só pra visualização
    else:
        model.fit(X_M, z)

    plot_decision_boundaries(model, X_M, z, nome)

plt.show()

# ==============================
#  FUNÇÕES AUXILIARES
# ==============================
def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(y_true == y_pred)

# ==============================
#  ESCOLHA DO LAMBDA IDEAL (K-FOLD)
# ==============================
def k_fold_friedman(X, y, lambdas, k=5):
    N = X.shape[0]
    indices = np.random.permutation(N)
    folds = np.array_split(indices, k)

    resultados = {}

    for lamb in lambdas:
        accs = []

        for i in range(k):
            test_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            clf = GaussianClassifierFriedman()
            clf.fit(X_train, y_train, lamb=lamb)

            y_pred = clf.predict(X_test)
            accs.append(accuracy_score(y_test, y_pred))

        resultados[lamb] = np.mean(accs)

    return resultados

lambdas = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5,
           0.6, 0.7, 0.8, 0.9, 1]

resultados_lambda = k_fold_friedman(X_M, z, lambdas, k=5)
melhor_lambda = max(resultados_lambda, key=resultados_lambda.get)

print("\n" + "=" * 90)
print("SELEÇÃO DO LAMBDA IDEAL - FRIEDMAN")
print("=" * 90)
for lamb, acc in resultados_lambda.items():
    print(f"lambda = {lamb:<6} | acurácia média = {acc:.4f}")

print(f"\nMelhor lambda: {melhor_lambda}")
print(f"Acurácia média do melhor lambda: {resultados_lambda[melhor_lambda]:.4f}")

# =========================================================
#  VALIDAÇÃO VIA MONTE CARLO
# =========================================================
# Aqui o foco é:
# - R = 500 rodadas
# - 80% treino / 20% teste
# - calcular a acurácia em cada rodada
# - armazenar cada acurácia em uma lista

def validation_monte_carlo(model_class, X, y, rodadas=500):
    N = X.shape[0]
    corte = int(0.8 * N)

    accuracy_list = []   # lista que armazena a acurácia de cada rodada

    for _ in range(rodadas):
        indices = np.random.permutation(N)

        train_idx = indices[:corte]
        test_idx = indices[corte:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = model_class()

        if model_class == GaussianClassifierFriedman:
            model.fit(X_train, y_train, lamb=melhor_lambda)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # cada acurácia obtida é guardada na lista
        accuracy_list.append(acc)

    return accuracy_list

# =========================================================
#  CÁLCULO DAS ESTATÍSTICAS FINAIS
# =========================================================
# Aqui o foco é:
# - usar a lista de acurácias
# - calcular média, desvio-padrão, maior e menor valor
# - colocar em tabela

print("\n" + "=" * 110)
print("TABELA FINAL - CLASSIFICAÇÃO")
print("=" * 110)
print(f"{'Modelo':<65} {'Média':<10} {'Desvio-Padrão':<15} {'Maior Valor':<12} {'Menor Valor':<12}")

resultados_finais = {}

for model_class, nome in modelos:
    accs = validation_monte_carlo(model_class, X_M, z, rodadas=500)

    media = np.mean(accs)
    desvio = np.std(accs)
    maior = np.max(accs)
    menor = np.min(accs)

    resultados_finais[nome] = {
        'accs': accs,
        'media': media,
        'desvio': desvio,
        'maior': maior,
        'menor': menor
    }

    print(f"{nome:<65} {media:<10.4f} {desvio:<15.4f} {maior:<12.4f} {menor:<12.4f}")

# ==============================
#  GRÁFICO DE BARRAS
# ==============================
nomes_modelos = list(resultados_finais.keys())
medias = [resultados_finais[n]['media'] for n in nomes_modelos]
desvios = [resultados_finais[n]['desvio'] for n in nomes_modelos]

plt.figure(figsize=(12, 6))
plt.bar(nomes_modelos, medias, yerr=desvios, capsize=5)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Acurácia")
plt.title("Comparação dos Modelos de Classificação")
plt.tight_layout()
plt.show()

# ==============================
#  BOXPLOT DAS ACURÁCIAS
# ==============================
dados_boxplot = [resultados_finais[n]['accs'] for n in nomes_modelos]

plt.figure(figsize=(12, 6))
plt.boxplot(dados_boxplot, tick_labels=nomes_modelos)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Acurácia")
plt.title("Distribuição das Acurácias por Modelo")
plt.tight_layout()
plt.show()