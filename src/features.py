"""
Paso 4: Construir features rezagadas para horizonte de 3 meses.
"""
import pandas as pd

# --- Cargar datos limpios ---
df = pd.read_csv("data/processed/datos_limpios.csv", parse_dates=["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

print(f"Datos cargados: {len(df)} filas")
print(f"Rango: {df['fecha'].min()} a {df['fecha'].max()}")

# --- Definir horizonte ---
HORIZONTE = 3  # predecir 3 meses adelante

# --- Crear features rezagadas ---
# Cada variable se rezaga al menos HORIZONTE meses
# Asi solo usamos informacion disponible al momento de predecir

rezagos = [3, 4, 5, 6, 12]

for lag in rezagos:
    df[f"inflacion_lag{lag}"] = df["inflacion_mensual"].shift(lag)
    df[f"tc_lag{lag}"] = df["tipo_cambio"].shift(lag)

for lag in rezagos:
    df[f"expect_lag{lag}"] = df["expectativas_inflacion"].shift(lag)

# --- Media movil de inflacion (ventana de 12 meses, rezagada al horizonte) ---
df["inflacion_ma12"] = df["inflacion_mensual"].shift(HORIZONTE).rolling(12).mean()

# --- Mes del anio (estacionalidad) ---
df["mes"] = df["fecha"].dt.month

# --- Variable objetivo adelantada ---
# No adelantamos el objetivo: predecimos inflacion_mensual
# Las features ya estan rezagadas, eso es equivalente

# --- Eliminar filas sin features completas ---
df_features = df.dropna().reset_index(drop=True)

print(f"\nFilas con features completas: {len(df_features)}")
print(f"Rango util: {df_features['fecha'].min()} a {df_features['fecha'].max()}")
print(f"Features creadas: {[c for c in df_features.columns if c not in ['fecha', 'inflacion_mensual', 'ipc_indice']]}")

# --- Guardar ---
df_features.to_csv("data/processed/features.csv", index=False)
print(f"\nGuardado en data/processed/features.csv")

# --- Verificacion: las features no deben correlacionar perfectamente con el objetivo ---
print("\nCorrelacion de cada feature con inflacion_mensual:")
cols_features = [c for c in df_features.columns if c not in ["fecha", "inflacion_mensual", "ipc_indice", "tipo_cambio", "expectativas_inflacion"]]
corr = df_features[cols_features + ["inflacion_mensual"]].corr()["inflacion_mensual"].drop("inflacion_mensual").sort_values(ascending=False)
print(corr.to_string())