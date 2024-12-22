import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
import joblib
import uuid
import time  # Importar para medir el tiempo
from scipy.stats import uniform, randint  # Para distribuciones de probabilidad
from sklearn.metrics import make_scorer, precision_score  # Para precisión ponderada

# 1. Cargar datos
ruta_csv = './Clustering/data_clustering.csv'  # Cambia a la ruta de tu archivo CSV
datos = pd.read_csv(ruta_csv)

# 2. Preparar datos
X = datos[['Forks', 'Contributors', 'Files', 'complexity', 'classes', 
           'duplicated_lines', 'duplicated_blocks', 'file_complexity', 
           'comment_lines_density', 'bugs']]
y = datos['Cluster']

# 3. Definir modelo base y nuevos parámetros para RandomizedSearch
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

# 5. Configurar RandomizedSearchCV con precisión ponderada
precision_weighted = make_scorer(precision_score, average='weighted')

# 6. Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=svm, param_distributions=parametros, 
                                   n_iter=100, cv=cv, scoring=precision_weighted, 
                                   verbose=1, n_jobs=-1, random_state=42)

# 7. Medir tiempo de optimización
print("Iniciando la búsqueda de hiperparámetros...")
inicio = time.time()  # Tiempo inicial
random_search.fit(X, y)
fin = time.time()  # Tiempo final

# Calcular tiempo en milisegundos
tiempo_total = (fin - inicio) * 1000  # Convertir a milisegundos
print(f"Búsqueda completada en {tiempo_total:.2f} ms.")

# 8. Obtener mejores parámetros y precisión
mejores_parametros = random_search.best_params_
mejor_precision = random_search.best_score_
print(f"Mejores parámetros: {mejores_parametros}")
print(f"Mejor precisión ponderada: {mejor_precision}")

# 9. Contar cuántos modelos se crearon durante la optimización
total_modelos = random_search.n_iter
print(f"Total de modelos creados: {total_modelos}")

# 10. Guardar el modelo entrenado
unique_id = str(uuid.uuid4().hex[:8])  # Generar identificador único más corto
nombre_modelo = f'./Desarrollo_Modelos/Modelo/modelo_svm_{unique_id}.pkl'
joblib.dump(random_search.best_estimator_, nombre_modelo)
print(f"Modelo guardado en: {nombre_modelo}")
