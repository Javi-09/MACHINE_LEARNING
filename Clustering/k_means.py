import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Cargar el archivo CSV
def load_data():
    try:
        file_path = './Limpieza_Data/data_limpio_final.csv'  # Ruta especificada del archivo CSV
        data = pd.read_csv(file_path)
        print(f"Datos cargados exitosamente desde {file_path}")
        return data
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

# Aplicar K-Means y generar un nuevo archivo CSV y una imagen
def apply_kmeans(data, n_clusters, output_file):
    try:
        # Seleccionar características numéricas para el clustering
        numeric_data = data.select_dtypes(include=['float64', 'int64'])

        # Estandarizar los datos con Min-Max Scaling
        scaler = MinMaxScaler()
        numeric_data_scaled = scaler.fit_transform(numeric_data)

        # Aplicar K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(numeric_data_scaled)

        # Agregar los clusters a los datos originales
        data['Cluster'] = clusters

        # Guardar los datos con los clusters en un nuevo archivo CSV
        data.to_csv(output_file, index=False)
        print(f"Archivo generado exitosamente: {output_file}")

        # Generar una gráfica de los clústeres si hay al menos dos columnas numéricas
        if numeric_data.shape[1] >= 2:
            # Usar las primeras dos columnas numéricas para la visualización
            plt.figure(figsize=(8, 6))
            plt.scatter(numeric_data_scaled[:, 0], numeric_data_scaled[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
            plt.title('Visualización de Clústeres')
            plt.xlabel(numeric_data.columns[0])
            plt.ylabel(numeric_data.columns[1])
            plt.colorbar(label='Cluster')

            # Guardar la gráfica en la misma ubicación que el CSV
            image_file = output_file.replace('.csv', '.png')
            plt.savefig(image_file)
            print(f"Gráfica de clústeres guardada como: {image_file}")
        else:
            print("No se puede generar la gráfica: se necesitan al menos dos columnas numéricas.")
    except Exception as e:
        print(f"Error al aplicar K-Means: {e}")

# Main
if __name__ == "__main__":
    # Archivo de salida
    output_file = './Clustering/data_clustering.csv'  # Ruta especificada del archivo de salida

    # Número de clusters
    n_clusters = 4

    # Cargar datos
    data = load_data()

    if data is not None:
        # Aplicar K-Means y guardar los resultados
        apply_kmeans(data, n_clusters, output_file)
