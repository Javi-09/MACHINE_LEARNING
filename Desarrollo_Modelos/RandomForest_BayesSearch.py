import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
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

# 3. Configurar modelo base
modelo_rf = RandomForestClassifier(random_state=42)

# 4. Definir los hiperparámetros para la optimización bayesiana con valores discretos
param_grid = {
    'n_estimators': [10, 100, 200, 500],             # Valores discretos para el número de árboles
    'max_depth': [None, 3, 4, 5, 10, 20, 50],         # Valores discretos para la profundidad máxima
    'min_samples_split': [2, 5, 10],                  # Valores discretos para división mínima de nodos
    'min_samples_leaf': [1, 2, 4],                    # Valores discretos para hojas mínimas por nodo
    'max_features': [None, 'sqrt', 'log2'],         # Valores discretos para max_features
    'bootstrap': [True, False],                       # Valores discretos para bootstrap
    'criterion': ['gini', 'entropy', 'log_loss']     # Valores discretos para criterio de división
}

# 5. Validación cruzada k-fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 6. Configurar BayesSearchCV
optimizador_bayesiano = BayesSearchCV(
    estimator=modelo_rf,
    search_spaces=param_grid,
    cv=cv,
    scoring='precision_weighted',  # Cambiar a precisión ponderada
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# 7. Medir el tiempo de optimización
start_time = time.time()  # Tiempo de inicio

# 8. Ajustar el modelo
print("Iniciando la búsqueda de hiperparámetros...")
optimizador_bayesiano.fit(X, y)
print("Búsqueda completada.")

# 9. Medir el tiempo total en milisegundos
end_time = time.time()  # Tiempo de finalización
tiempo_total = (end_time - start_time) * 1000  # Convertir a milisegundos
print(f"Tiempo total de optimización: {tiempo_total:.2f} ms")

# 10. Obtener mejores parámetros y precisión
mejores_parametros = optimizador_bayesiano.best_params_
mejor_precision = optimizador_bayesiano.best_score_
print(f"Mejores parámetros: {mejores_parametros}")
print(f"Mejor precisión: {mejor_precision}")

# 11. Entrenar el modelo con los mejores parámetros encontrados
mejor_modelo = optimizador_bayesiano.best_estimator_

# 12. Generar un identificador único corto (UUID)
unique_id = str(uuid.uuid4().hex[:8])  # Solo tomamos los primeros 8 caracteres para hacerlo corto

# Crear el nombre del archivo del modelo con el identificador
nombre_modelo = f'./Desarrollo_Modelos/Modelo/modelo_rf_{unique_id}.pkl'

# 13. Guardar el modelo optimizado
joblib.dump(mejor_modelo, nombre_modelo)
print(f"Modelo optimizado guardado como: {nombre_modelo}")

# 14. Contar el número de modelos creados durante la optimización
num_modelos = len(optimizador_bayesiano.cv_results_['mean_test_score'])
print(f"Número de modelos creados durante la optimización: {num_modelos}")
