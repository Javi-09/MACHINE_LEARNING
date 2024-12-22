import pandas as pd
import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
import joblib
import uuid
import time

# 1. Cargar datos
ruta_csv = './Clustering/data_clustering.csv'  # Cambia a la ruta de tu archivo CSV
datos = pd.read_csv(ruta_csv)

# 2. Preparar datos (solo las columnas especificadas)
X = datos[['Forks', 'Contributors', 'Files', 'complexity', 'classes', 
           'duplicated_lines', 'duplicated_blocks', 'file_complexity', 
           'comment_lines_density', 'bugs']]
y = datos['Cluster']

# Verificar si hay clases con muy pocos elementos
clases_con_pocos_elementos = y.value_counts()[y.value_counts() < 10]
if not clases_con_pocos_elementos.empty:
    print("Advertencia: Existen clases con menos de 10 elementos. Esto puede afectar el rendimiento del modelo.")
    print(clases_con_pocos_elementos)

# 3. Definir función objetivo para PSO
total_modelos_creados = 0  # Contador para el número total de modelos creados

def funcion_objetivo(params):
    global total_modelos_creados
    scores = []
    for param_set in params:
        n_estimators = int(param_set[0])
        max_depth = int(param_set[1]) if param_set[1] > 0 else None
        min_samples_split = int(param_set[2])
        min_samples_leaf = int(param_set[3])
        max_features = ['sqrt', 'log2', None][int(param_set[4])]  # 'auto' eliminado
        bootstrap = bool(round(param_set[5]))
        criterion = ['gini', 'entropy', 'log_loss'][int(param_set[6])]

        # Crear el modelo RandomForest con los parámetros actuales
        modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            criterion=criterion,
            random_state=42
        )

        # Validación cruzada
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Entrenar y predecir
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            fold_scores.append(precision_score(y_test, y_pred, average='weighted'))

        scores.append(-np.mean(fold_scores))  # Negativo para minimización
        total_modelos_creados += 1  # Contar cada modelo creado
    return np.array(scores)

# 4. Definir los límites de los parámetros para PSO
limites = np.array([
    [10, 500],         # n_estimators
    [1, 50],           # max_depth
    [2, 10],           # min_samples_split
    [1, 4],            # min_samples_leaf
    [0, 2],            # max_features (índices: sqrt, log2, None)
    [0, 1],            # bootstrap (0: False, 1: True)
    [0, 2]             # criterion (índices: gini, entropy, log_loss)
]).T

# 5. Configurar y ejecutar PSO
optimizer = GlobalBestPSO(
    n_particles=10,  # Tamaño del enjambre
    dimensions=7,    # Número de hiperparámetros a optimizar
    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},  # Parámetros del PSO
    bounds=(limites[0], limites[1])  # Límites inferior y superior
)

# 6. Medir tiempo de optimización
print("Iniciando la optimización por PSO...")
inicio = time.time()
cost, pos = optimizer.optimize(funcion_objetivo, iters=50)
fin = time.time()

# Calcular tiempo en milisegundos
tiempo_total = (fin - inicio) * 1000
print(f"Optimización completada en {tiempo_total:.2f} ms.")

# 7. Obtener los mejores parámetros
n_estimators_opt = int(pos[0])
max_depth_opt = int(pos[1]) if pos[1] > 0 else None
min_samples_split_opt = int(pos[2])
min_samples_leaf_opt = int(pos[3])
max_features_opt = ['sqrt', 'log2', None][int(pos[4])]
bootstrap_opt = bool(round(pos[5]))
criterion_opt = ['gini', 'entropy', 'log_loss'][int(pos[6])]

print("Mejores parámetros:")
print(f"n_estimators = {n_estimators_opt}")
print(f"max_depth = {max_depth_opt}")
print(f"min_samples_split = {min_samples_split_opt}")
print(f"min_samples_leaf = {min_samples_leaf_opt}")
print(f"max_features = {max_features_opt}")
print(f"bootstrap = {bootstrap_opt}")
print(f"criterion = {criterion_opt}")

# 8. Entrenar modelo final con los mejores parámetros y calcular la precisión con validación cruzada
modelo_final = RandomForestClassifier(
    n_estimators=n_estimators_opt,
    max_depth=max_depth_opt,
    min_samples_split=min_samples_split_opt,
    min_samples_leaf=min_samples_leaf_opt,
    max_features=max_features_opt,
    bootstrap=bootstrap_opt,
    criterion=criterion_opt,
    random_state=42
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
precision_scores = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    modelo_final.fit(X_train, y_train)
    y_pred = modelo_final.predict(X_test)
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))

precision_promedio = np.mean(precision_scores)
#print(f"Precisión ponderada promedio del modelo final en validación cruzada: {precision_promedio:.4f}")
print(f"Precisión ponderada promedio del modelo final en validación cruzada: {precision_promedio}")


# 9. Guardar el modelo
unique_id = str(uuid.uuid4().hex[:8])
nombre_modelo = f'./Desarrollo_Modelos/Modelo/modelo_rf_{unique_id}.pkl'
joblib.dump(modelo_final, nombre_modelo)
print(f"Modelo optimizado guardado como: {nombre_modelo}")

# 10. Mostrar el número total de modelos creados
print(f"Total de modelos creados durante la optimización: {total_modelos_creados}")
