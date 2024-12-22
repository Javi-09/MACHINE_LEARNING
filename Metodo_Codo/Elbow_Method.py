import matplotlib
matplotlib.use('Agg')  # Establecer el backend a Agg para evitar el problema con Tcl/Tk

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar los datos desde el archivo CSV
df = pd.read_csv('./Limpieza_Data/data_limpio_final.csv')

# Seleccionar solo las columnas numéricas para el clustering
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Aplicar KMeans para varios valores de K (número de clústeres)
inertia = []
for k in range(1, 21):  # Usamos un rango de 1 a 20 clústeres
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_numeric)
    inertia.append(kmeans.inertia_)

# Graficar el Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 21), inertia, marker='o')  # Graficamos con el rango de 1 a 20
plt.title('Elbow Method For Optimal k')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.grid(True)

# Ajustar los ticks del eje X para que sean enteros del 1 al 20
plt.xticks(range(1, 21))  # Establecer los valores de los ticks en el eje X

# Guardar la gráfica como archivo PNG
plt.savefig('./Metodo_Codo/elbow_method_graph.png')

print("Gráfica guardada como 'elbow_method_graph.png'")
