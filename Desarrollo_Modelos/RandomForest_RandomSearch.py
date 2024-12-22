import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
import uuid  # Para generar identificadores únicos
import time  # Para medir el tiempo de optimización

# 1. Cargar datos
ruta_csv = './Clustering/data_clustering.csv'  # Cambia a la ruta de tu archivo CSV
datos = pd.read_csv(ruta_csv)

# 2. Preparar datos
X = datos[['Forks', 'Contributors', 'Files', 'complexity', 'classes', 
           'duplicated_lines', 'duplicated_blocks', 'file_complexity', 
           'comment_lines_density', 'bugs']]  # Variables seleccionadas
y = datos['Cluster']

# 3. Configurar el modelo RandomForestClassifier
modelo_rf = RandomForestClassifier(random_state=42)

# 4. Definir los hiperparámetros para la optimización con RandomizedSearchCV
param_dist = {
    'n_estimators': [10, 100, 200, 500],
    'max_depth': [None, 3, 4, 5, 10, 20, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy', 'log_loss']
}

# 5. Configurar RandomizedSearchCV con validación cruzada 10-fold y métrica de precisión ponderada (precision_weighted)
random_search = RandomizedSearchCV(
    estimator=modelo_rf, 
    param_distributions=param_dist, 
    n_iter=100, 
    cv=10, 
    scoring='precision_weighted', 
    n_jobs=-1, 
    random_state=42
)

# 6. Medir el tiempo de optimización
start_time = time.time()

# 7. Ejecutar la búsqueda aleatoria (Random Search)
random_search.fit(X, y)

# 8. Obtener el número de modelos evaluados
contador_modelos = random_search.n_iter

# 9. Medir el tiempo de optimización en milisegundos
end_time = time.time()
optimization_time = (end_time - start_time) * 1000  # en milisegundos

# 10. Mostrar los mejores parámetros encontrados y el mejor score
print(f"Mejores parámetros: {random_search.best_params_}")
print(f"Mejor score (precision_weighted): {random_search.best_score_}")

# 11. Mostrar el tiempo total de optimización en milisegundos
print(f"Tiempo de optimización: {optimization_time:.2f} ms")

# 12. Entrenar el modelo con los mejores parámetros encontrados
mejor_modelo = random_search.best_estimator_

# 13. Generar un identificador único corto (UUID)
unique_id = str(uuid.uuid4().hex[:8])  # Solo tomamos los primeros 8 caracteres para hacerlo corto

# Crear el nombre del archivo del modelo con el identificador
nombre_modelo = f'./Desarrollo_Modelos/Modelo/modelo_rf_{unique_id}.pkl'

# 14. Guardar el modelo optimizado
joblib.dump(mejor_modelo, nombre_modelo)

# 15. Mostrar información del modelo creado
print(f"Modelo optimizado guardado como: {nombre_modelo}")
print(f"Número total de modelos evaluados: {contador_modelos}")
