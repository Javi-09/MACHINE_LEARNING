import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

# Cambiar backend
matplotlib.use('Agg')  # Usa un backend no interactivo

# Cargar el archivo CSV
file_path = './Clustering/data_clustering.csv'  # Ruta del archivo CSV
data = pd.read_csv(file_path)

# Variables independientes y dependiente
X = data[['Forks', 'Commits', 'Open Issues', 'Contributors', 'Files', 'ncloc', 'complexity', 
          'cognitive_complexity', 'classes', 'functions', 'duplicated_lines', 'duplicated_blocks', 
          'comment_lines', 'file_complexity', 'comment_lines_density', 'bugs']]
y = data['Cluster']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar el modelo Ridge con validación cruzada
ridge = RidgeCV(cv=5).fit(X_train, y_train)

# Coeficientes del modelo
coefficients = ridge.coef_

# Identificar variables relevantes e irrelevantes
relevant_vars = []
irrelevant_vars = []

for var, coef in zip(X.columns, coefficients):
    if coef != 0:
        relevant_vars.append(var)
    else:
        irrelevant_vars.append(var)

# Mostrar resultados
print("Variables relevantes:", relevant_vars)
print("Variables irrelevantes:", irrelevant_vars)

# Crear un gráfico de las magnitudes de los coeficientes
graph_path = './Regularizacion/ridge_coefficients.png'  # Ruta donde se guardará la gráfica
plt.figure(figsize=(10, 6))
plt.bar(X.columns, np.abs(coefficients), color="lightcoral")
plt.xticks(rotation=45, ha="right")
plt.title("Importancia de las variables según Ridge")
plt.ylabel("Magnitud del coeficiente")
plt.xlabel("Variables")
plt.tight_layout()
plt.savefig(graph_path)  # Guarda la gráfica en un archivo
print(f"La gráfica se guardó en {graph_path}")
