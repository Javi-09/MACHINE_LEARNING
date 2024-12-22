import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import joblib
import uuid
import time  # Importar para medir el tiempo
from sklearn.metrics import make_scorer, precision_score

# 1. Cargar datos
ruta_csv = './Clustering/data_clustering.csv'  # Cambia a la ruta de tu archivo CSV
datos = pd.read_csv(ruta_csv)

# 2. Preparar datos
X = datos[['Forks', 'Contributors', 'Files', 'complexity', 'classes', 
           'duplicated_lines', 'duplicated_blocks', 'file_complexity', 
           'comment_lines_density', 'bugs']]
y = datos['Cluster']

# 3. Definir modelo base y nuevos parámetros para GridSearch
svm = SVC()
parametros = {
    'C': [0.1, 1, 10, 50, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10],
    'degree': [2, 3, 4],  # Relevante para kernel 'poly'
    'coef0': [0.0, 0.01, 0.1, 0.5, 1, 10]  # Relevante para kernels 'poly' y 'sigmoid'
}

# 4. Validación cruzada k-fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 5. Configurar el scorer para precisión ponderada
precision_ponderada = make_scorer(precision_score, average='weighted')

# 6. Configurar GridSearchCV con la nueva métrica
grid_search = GridSearchCV(estimator=svm, param_grid=parametros, cv=cv, 
                           scoring=precision_ponderada, verbose=1, n_jobs=-1)

# 7. Medir tiempo de optimización
print("Iniciando la búsqueda de hiperparámetros...")
inicio = time.time()  # Tiempo inicial
grid_search.fit(X, y)
fin = time.time()  # Tiempo final

# Calcular tiempo en milisegundos
tiempo_total = (fin - inicio) * 1000  # Convertir a milisegundos
print(f"Búsqueda completada en {tiempo_total:.2f} ms.")

# 8. Obtener número total de modelos evaluados
num_modelos_evaluados = len(grid_search.cv_results_['mean_test_score'])
print(f"Número total de modelos evaluados: {num_modelos_evaluados}")

# 9. Obtener mejores parámetros y precisión ponderada
mejores_parametros = grid_search.best_params_
mejor_precision = grid_search.best_score_
print(f"Mejores parámetros: {mejores_parametros}")
print(f"Mejor precisión ponderada: {mejor_precision}")

# 10. Guardar el modelo entrenado
unique_id = str(uuid.uuid4().hex[:8])  # Generar identificador único más corto
nombre_modelo = f'./Desarrollo_Modelos/Modelo/modelo_svm_{unique_id}.pkl'
joblib.dump(grid_search.best_estimator_, nombre_modelo)
print(f"Modelo guardado en: {nombre_modelo}")
