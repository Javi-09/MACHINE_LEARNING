import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import precision_score, make_scorer
import joblib
import uuid
import time  # Para medir el tiempo de optimización

# 1. Cargar datos
ruta_csv = './Clustering/data_clustering.csv'  # Cambia a la ruta de tu archivo CSV
datos = pd.read_csv(ruta_csv)

# 2. Preparar datos
X = datos[['Forks', 'Contributors', 'Files', 'complexity', 'classes', 
           'duplicated_lines', 'duplicated_blocks', 'file_complexity', 
           'comment_lines_density', 'bugs']]
y = datos['Cluster']

# 3. Definir modelo base y parámetros para optimización bayesiana
svm = SVC()

parametros = {
    'C': [0.1, 1, 10, 50, 100, 1000],  # Valores específicos para C
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Tipos de kernel
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10],  # Valores de gamma
    'degree': [2, 3, 4],  # Grados para el kernel 'poly'
    'coef0': [0.0, 0.01, 0.1, 0.5, 1, 10]  # Coef0 para los kernels 'poly' y 'sigmoid'
}

# 4. Validación cruzada k-fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 5. Métrica de precisión ponderada
precision_ponderada = make_scorer(precision_score, average='weighted')

# 6. Configurar BayesSearchCV
optimizador_bayesiano = BayesSearchCV(
    estimator=svm, 
    search_spaces=parametros, 
    cv=cv, 
    scoring=precision_ponderada, 
    verbose=1, 
    n_jobs=-1
)

# 7. Contador de modelos creados
contador_modelos = 0
def callback(optim_result):
    global contador_modelos
    contador_modelos += 1

# 8. Medir el tiempo de optimización
start_time = time.time()  # Tiempo de inicio

# 9. Ajustar el modelo
print("Iniciando la búsqueda de hiperparámetros...")
optimizador_bayesiano.fit(X, y, callback=callback)
print("Búsqueda completada.")

# 10. Medir el tiempo total en milisegundos
end_time = time.time()  # Tiempo de finalización
tiempo_total = (end_time - start_time) * 1000  # Convertir a milisegundos
print(f"Tiempo total de optimización: {tiempo_total:.2f} ms")

# 11. Obtener mejores parámetros y precisión
mejores_parametros = optimizador_bayesiano.best_params_
mejor_precision = optimizador_bayesiano.best_score_
print(f"Mejores parámetros: {mejores_parametros}")
print(f"Mejor precisión ponderada: {mejor_precision}")

# 12. Guardar el modelo entrenado
unique_id = str(uuid.uuid4().hex[:8])  # Generar identificador único más corto
nombre_modelo = f'./Desarrollo_Modelos/Modelo/modelo_svm_{unique_id}.pkl'
joblib.dump(optimizador_bayesiano.best_estimator_, nombre_modelo)
print(f"Modelo guardado en: {nombre_modelo}")

# 13. Mostrar el contador de modelos
print(f"Total de modelos creados durante la optimización: {contador_modelos}")
