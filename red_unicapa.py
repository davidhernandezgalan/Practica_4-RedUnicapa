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

# Valores deseados 
ax.scatter(range(len(D)), D[:, 0], color=colors['d1']['deseado'], 
           label='Deseado d1', alpha=1, s=130, edgecolors='white', linewidths=2)
ax.scatter(np.array(range(len(D))) + desplazamiento, D[:, 1], 
           color=colors['d2']['deseado'], label='Deseado d2', alpha=1, 
           s=130, edgecolors='white', linewidths=2)
ax.scatter(range(len(D)), D[:, 2], color=colors['d3']['deseado'], 
           label='Deseado d3', alpha=1, s=130, edgecolors='white', linewidths=2)

# Predicciones 
ax.scatter(range(len(salida_entrenada)), salida_entrenada[:, 0], 
           color=colors['d1']['prediccion'], marker='D', 
           label='Predicción d1', alpha=0.8, s=100, linewidth=1.5)
ax.scatter(np.array(range(len(salida_entrenada))) + desplazamiento, 
           salida_entrenada[:, 1], color=colors['d2']['prediccion'], 
           marker='D', label='Predicción d2', alpha=0.8, s=100, linewidth=1.5)
ax.scatter(range(len(salida_entrenada)), salida_entrenada[:, 2], 
           color=colors['d3']['prediccion'], marker='D', 
           label='Predicción d3', alpha=0.8, s=100, linewidth=1.5)

# Configuración de ejes y bordes
ax.set_facecolor('black')
for spine in ax.spines.values():
    spine.set_color('white')
    spine.set_linewidth(1.5)

# Ajustes de ejes y etiquetas
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('', color='white', fontsize=12, labelpad=10)
ax.set_ylabel('Activación (0 o 1)', color='white', fontsize=12, labelpad=10)
ax.tick_params(colors='white', which='both', labelsize=10)

# Leyenda 
legend = ax.legend(facecolor='#222222', edgecolor='white', labelcolor='white',
                  fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left',
                  title='Resultados', title_fontsize=11)
legend.get_title().set_color('white')

# Título
plt.title('Comparación: Salidas Deseadas vs Predicciones de la Red Neuronal', 
          color='white', fontsize=14, pad=20)

plt.tight_layout()
plt.show(block=False)

# Paso 4: Matrices de confusión 
plt.figure(facecolor='black', figsize=(14, 5.5))  # Tamaño optimizado

# Crear una disposición de cuadrícula para organizar los elementos
grid = plt.GridSpec(2, 3, height_ratios=[0.1, 1], hspace=0.3, wspace=0.4)

# Crear los ejes para las matrices
ax1 = plt.subplot(grid[1, 0])
ax2 = plt.subplot(grid[1, 1])
ax3 = plt.subplot(grid[1, 2])
axs = [ax1, ax2, ax3]

# Eje para la barra de color 
cbar_ax = plt.subplot(grid[0, :])

# Colormap para matrices
cmap = 'viridis'
vmin = min([confusion_matrix(D[:, i], salida_entrenada_binaria[:, i]).min() for i in range(3)])
vmax = max([confusion_matrix(D[:, i], salida_entrenada_binaria[:, i]).max() for i in range(3)])

for i in range(3):
    matriz_confusion = confusion_matrix(D[:, i], salida_entrenada_binaria[:, i])
    im = axs[i].matshow(matriz_confusion, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Configuración de títulos y etiquetas
    axs[i].set_title(f'Matriz d{i+1}', color='white', fontsize=12, pad=10)
    axs[i].set_xlabel('Predicción', color='white', fontsize=10)
    axs[i].set_ylabel('Real', color='white', fontsize=10)
    axs[i].xaxis.set_ticks_position('bottom')
    axs[i].tick_params(colors='white', labelsize=9)
    
    # Grid en matrices
    axs[i].grid(which='major', color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Etiquetas en celdas 
    for (j, k), value in np.ndenumerate(matriz_confusion):
        axs[i].text(k, j, f'{value}', ha='center', va='center', 
                   color='white', fontsize=12, fontweight='bold')

# Añadir barra de color
cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(colors='white')
cbar.ax.xaxis.set_tick_params(color='white')
plt.setp(cbar.ax.get_xticklabels(), color='white')
cbar_ax.set_title('Escala de Valores', color='white', pad=5)

# Ajustar el layout para evitar solapamientos
plt.tight_layout()

plt.show()