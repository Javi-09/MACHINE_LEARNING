import pandas as pd

#IMPUTACION - MEDIANA
#ELIMINAR COLUMNA NONE SUPERA 50%

# Cargar el archivo CSV
df = pd.read_csv('./Data/METRICAS_GH_SQ_FINAL.csv')

# Listas para almacenar las columnas modificadas
columns_imputed = []
columns_dropped = []

# Iterar sobre las columnas
for col in df.columns:
    # Verificar si la columna contiene valores faltantes
    if df[col].isnull().sum() > 0:
        # Calcular el porcentaje de valores faltantes
        missing_percentage = df[col].isnull().mean() * 100
        
        # Si el porcentaje de valores faltantes es mayor al 50%, eliminar la columna
        if missing_percentage > 50:
            print(f"Columna '{col}' tiene más del 50% de valores faltantes y será eliminada.")
            columns_dropped.append(col)
            df = df.drop(columns=[col])
        else:
            # Si no supera el 50%, imputar con la mediana
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Columna '{col}' fue imputada con la mediana.")
            columns_imputed.append(col)

# Mostrar las columnas que fueron modificadas
print("\nColumnas a las que se les aplicó la mediana:", columns_imputed)
print("Columnas eliminadas por superar el 50% de valores faltantes:", columns_dropped)

# Guardar el DataFrame limpio en un nuevo archivo CSV
df.to_csv('./Limpieza_Data/data_limpio.csv', index=False)

print("El proceso de limpieza e imputación ha terminado. Los datos se guardaron en 'Limpieza_Data/data_limpio.csv'.")
