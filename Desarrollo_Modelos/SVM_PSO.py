import pandas as pd
import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
import joblib
import uuid
import time

# 1. Cargar datos
ruta_csv = './Clustering/data_clustering.csv'  # Cambia a la ruta de tu archivo CSV
datos = pd.read_csv(ruta_csv)

# 2. Preparar datos
X = datos[['Forks', 'Contributors', 'Files', 'complexity', 'classes', 
           'duplicated_lines', 'duplicated_blocks', 'file_complexity', 
           'comment_lines_density', 'bugs']]
y = datos['Cluster']

# Contador de modelos evaluados
contador_modelos = 0

# 3. Definir función objetivo para PSO
def funcion_objetivo(params):
    global contador_modelos
    # Cada fila de params representa un conjunto de hiperparámetros
    scores = []

    for param_set in params:
        contador_modelos += 1  # Incrementar contador por cada modelo evaluado
        C = param_set[0]
        kernel = ['linear', 'poly', 'rbf', 'sigmoid'][int(param_set[1])]  # Convertir a string para el kernel
        gamma = ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10][int(param_set[2])]  # Convertir gamma
        degree = int(param_set[3])
        coef0 = [0.0, 0.01, 0.1, 0.5, 1, 10][int(param_set[4])]  # Convertir coef0

        # Crear el modelo SVM con los parámetros actuales
        svm = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

        # Validación cruzada
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Entrenar y predecir
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            fold_scores.append(precision_score(y_test, y_pred, average='weighted'))  # Usar precisión ponderada

        # Guardar el promedio de precisión de este conjunto de parámetros
        scores.append(-np.mean(fold_scores))  # Negativo para minimización

    return np.array(scores)

# 4. Definir los límites de los parámetros para PSO
# Los límites de los parámetros son: C, kernel, gamma, degree y coef0
limites = np.array([
    [0.1, 1000],       # Límites para C
    [0, 3],            # Límites para kernel (0: 'linear', 1: 'poly', 2: 'rbf', 3: 'sigmoid')
    [0, 7],            # Límites para gamma (índices de la lista de valores)
    [2, 4],            # Límites para degree
    [0, 5]             # Límites para coef0
]).T

# 5. Configurar y ejecutar PSO
optimizer = GlobalBestPSO(
    n_particles=10,  # Tamaño del enjambre
    dimensions=5,    # Número de hiperparámetros a optimizar
    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},  # Parámetros del PSO
    bounds=(limites[0], limites[1])  # Límites inferior y superior
)

# 6. Medir tiempo de optimización
print("Iniciando la optimización por PSO...")
inicio = time.time()  # Tiempo inicial
cost, pos = optimizer.optimize(funcion_objetivo, iters=10)
fin = time.time()  # Tiempo final

# Calcular tiempo en milisegundos
tiempo_total = (fin - inicio) * 1000  # Convertir a milisegundos
print(f"Optimización completada en {tiempo_total:.2f} ms.")

# 7. Obtener los mejores parámetros
C_optimo = pos[0]
kernel_optimo = ['linear', 'poly', 'rbf', 'sigmoid'][int(pos[1])]
gamma_optimo = ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10][int(pos[2])]
degree_optimo = int(pos[3])
coef0_optimo = [0.0, 0.01, 0.1, 0.5, 1, 10][int(pos[4])]

# Calcular la precisión final
mejor_precision = -cost  # Convertir de vuelta a precisión positiva

print(f"Mejores parámetros: C = {C_optimo}, kernel = {kernel_optimo}, gamma = {gamma_optimo}, degree = {degree_optimo}, coef0 = {coef0_optimo}")
print(f"Mejor precisión ponderada: {mejor_precision}")
print(f"Total de modelos evaluados: {contador_modelos}")

# 8. Entrenar modelo final con los mejores parámetros
svm_final = SVC(C=C_optimo, kernel=kernel_optimo, gamma=gamma_optimo, degree=degree_optimo, coef0=coef0_optimo)
svm_final.fit(X, y)

# 9. Guardar el modelo
unique_id = str(uuid.uuid4().hex[:8])  # Generar identificador único más corto
nombre_modelo = f'./Desarrollo_Modelos/Modelo/modelo_svm_{unique_id}.pkl'
joblib.dump(svm_final, nombre_modelo)
print(f"Modelo guardado en: {nombre_modelo}")

# 10. Guardar el escalador (importante para preprocesar datos nuevos)
nombre_escalador = f'./Desarrollo_Modelos/Modelo/escalador_svm_{unique_id}.pkl'
print(f"Nota: No se usó escalado en este caso.")
