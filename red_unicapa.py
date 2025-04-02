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

# Entrenamiento de la red neuronal
for epoch in range(epochs):
    # Forward propagation
    salida = forward_propagation(X, pesos, sesgo)
    
    # Calcular el error (deseado - predicción)
    error = D - salida
    
    # Retropropagación del error y ajuste de los pesos
    ajuste = error * sigmoide_derivada(salida)
    
    # Ajustar los pesos y el sesgo
    pesos += np.dot(X.T, ajuste) * learning_rate
    sesgo += np.sum(ajuste, axis=0, keepdims=True) * learning_rate
    
    # Imprimir el error promedio cada 1000 épocas
    if epoch % 1000 == 0:
        error_promedio = np.mean(np.abs(error))
        print(f"Época {epoch}: Error promedio: {error_promedio}")

# Paso 2: Ejecutar la red para obtener la salida final después del entrenamiento
salida_entrenada = forward_propagation(X, pesos, sesgo)

# Redondear las predicciones a 0 o 1
salida_entrenada_binaria = np.round(salida_entrenada)

# Paso 3: Visualización de las salidas deseadas vs predicciones
fig, ax = plt.subplots(figsize=(12, 7), facecolor='black')

# Configurar colores personalizados 
colors = {
    'd1': {'deseado': '#0066FF', 'prediccion': '#7fbfff'},  # Azul eléctrico
    'd2': {'deseado': '#FF6600', 'prediccion': '#ffb07f'},  # Naranja brillante
    'd3': {'deseado': '#CC00FF', 'prediccion': '#c5a3d8'}   # Morado neón
}

# Desplazamiento 
desplazamiento = 0.05

# Configuración de grid 
ax.grid(True, color='lightgray', linestyle='--', linewidth=0.8, alpha=0.6)

# Línea horizontal de referencia en 0.5
ax.axhline(y=0.5, color='white', linestyle=':', linewidth=1.2, alpha=0.7)
