import matplotlib
matplotlib.use('Agg')  # Establecer el backend a 'Agg' (sin GUI)

import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
file_path = './Data/METRICAS_GH_SQ_FINAL.csv'  # Asegúrate de que el archivo esté en el mismo directorio que el script o proporciona la ruta completa
data = pd.read_csv(file_path)

# Calcular estadísticas descriptivas
stats = data.describe().T  # Describir y transponer para mejor visualización
stats['range'] = stats['max'] - stats['min']  # Agregar columna de rango (máximo - mínimo)

# Mostrar las estadísticas en consola
print("Estadísticas descriptivas:\n", stats)

# Guardar las estadísticas en un nuevo archivo CSV
stats.to_csv('./Descripción_Estructural/estadisticas_descriptivas.csv', columns=['mean', 'std', 'min', 'max', 'range'])

# Crear gráficas para visualizar las métricas
plt.figure(figsize=(12, 8))

# Graficar las medias
plt.subplot(2, 2, 1)
stats['mean'].plot(kind='bar', color='skyblue')
plt.title('Media de las métricas')
plt.ylabel('Valor')
plt.xticks(rotation=45, ha='right')

# Graficar la desviación estándar
plt.subplot(2, 2, 2)
stats['std'].plot(kind='bar', color='lightcoral')
plt.title('Desviación Estándar de las métricas')
plt.ylabel('Valor')
plt.xticks(rotation=45, ha='right')

# Graficar el mínimo
plt.subplot(2, 2, 3)
stats['min'].plot(kind='bar', color='lightgreen')
plt.title('Mínimo de las métricas')
plt.ylabel('Valor')
plt.xticks(rotation=45, ha='right')

# Graficar el máximo
plt.subplot(2, 2, 4)
stats['max'].plot(kind='bar', color='orange')
plt.title('Máximo de las métricas')
plt.ylabel('Valor')
plt.xticks(rotation=45, ha='right')

# Ajustar la disposición de las gráficas
plt.tight_layout()

# Guardar las gráficas como archivos PNG
plt.savefig('./Descripción_Estructural/graficas_metrica.png')

# Mensaje indicando que las gráficas fueron guardadas
print("Las gráficas se han guardado como 'graficas_metrica.png'.")
