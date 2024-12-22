import pandas as pd
from sklearn.preprocessing import RobustScaler  # Importar RobustScaler para escalado robusto

# Cargar el CSV
df = pd.read_csv('./Limpieza_Data/data_limpio.csv')

# Inicializar el escalador RobustScaler
scaler = RobustScaler()

# Aplicar el escalado robusto a todas las columnas (excepto si alguna es categórica)
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Guardar el nuevo CSV con los datos escalados
df_scaled.to_csv('./Limpieza_Data/data_limpio_final.csv', index=False)

print("El archivo CSV ha sido escalado (Escalado Robusto) y guardado como 'data_limpio_final.csv'.")
