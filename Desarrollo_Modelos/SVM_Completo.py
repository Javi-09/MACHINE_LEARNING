# 1. Cargar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score
import joblib
import uuid
import hashlib

# 2. Cargar y preparar los datos
ruta_csv = './Clustering/data_clustering.csv'  # Cambia a la ruta de tu archivo CSV
datos = pd.read_csv(ruta_csv)

# Selección de características (X) y etiqueta (y)
X = datos[['Forks', 'Commits', 'Open Issues', 'Contributors', 'Files', 'ncloc',
           'complexity', 'cognitive_complexity', 'classes', 'functions', 
           'duplicated_lines', 'duplicated_blocks', 'comment_lines', 
           'file_complexity', 'comment_lines_density', 'bugs']]
y = datos['Cluster']

# 3. Ya tienes los datos escalados, por lo tanto no es necesario volver a escalarlos

# 4. Crear el modelo SVM (sin modificar los hiperparámetros por defecto)
svm_model = SVC()

# 5. Validación cruzada con 10 pliegues y múltiples métricas
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Definir las métricas
scoring = {'accuracy': make_scorer(accuracy_score), 
           'precision': make_scorer(precision_score, average='weighted')}  # Para clasificación multiclase

# Realizar validación cruzada
scores = cross_validate(svm_model, X, y, cv=cv, scoring=scoring)

# Mostrar los resultados de las métricas
print(f'Promedio de Accuracy: {np.mean(scores["test_accuracy"])}')
print(f'Promedio de Precision: {np.mean(scores["test_precision"])}')

# 6. Entrenar el modelo con todos los datos (ya escalados)
svm_model.fit(X, y)

# 7. Generar un identificador único más corto usando hash SHA-1
unique_id = hashlib.sha1(str(uuid.uuid4()).encode()).hexdigest()[:8]  # Tomar los primeros 8 caracteres

# 8. Crear el nombre del archivo del modelo
nombre_modelo = f'./Desarrollo_Modelos/Modelo/modelo_svm_{unique_id}.pkl'

# 9. Guardar el modelo entrenado
joblib.dump(svm_model, nombre_modelo)

# Mostrar la ruta donde se guardó el modelo
print(f'Modelo guardado en: {nombre_modelo}')
