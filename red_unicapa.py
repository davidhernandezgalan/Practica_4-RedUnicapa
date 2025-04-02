import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Configuración de estilo para fondo oscuro
plt.style.use('dark_background')

# Paso 1: Cargar los archivos
entradas = pd.read_csv(r'C:\Users\victo\Documents\Universidad\10° Semestre\Seminario de Solución de Problemas de Inteligencia Artificial 2\Practica_4-RedUnicapa\x.csv')
deseado = pd.read_csv(r'C:\Users\victo\Documents\Universidad\10° Semestre\Seminario de Solución de Problemas de Inteligencia Artificial 2\Practica_4-RedUnicapa\deseadoandorxor.csv')

# Convertir los datos de entrada y salida a matrices
X = entradas.values
D = deseado.values

# Parámetros de la red neuronal
np.random.seed(42)
pesos = np.random.uniform(-1, 1, (2, 3))  # 2 entradas, 3 salidas
sesgo = np.random.uniform(-1, 1, (1, 3))  # Sesgo para cada neurona
learning_rate = 0.5  # Tasa de aprendizaje
epochs = 10000  # Número de épocas

# Función de activación: Sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la sigmoide
def sigmoide_derivada(x):
    return x * (1 - x)

# Propagación hacia adelante (forward propagation)
def forward_propagation(X, pesos, sesgo):
    z = np.dot(X, pesos) + sesgo  # Producto punto entre entradas y pesos + sesgo
    return sigmoide(z)  # Aplicar activación
