"""
Prediccion real: proximo mes sin datos.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --- Cargar features ---
df = pd.read_csv("data/processed/features.csv", parse_dates=["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

# --- Definir columnas ---
OBJETIVO = "inflacion_mensual"
EXCLUIR = ["fecha", "inflacion_mensual", "ipc_indice", "tipo_cambio", "expectativas_inflacion"]
FEATURES = [c for c in df.columns if c not in EXCLUIR]

# --- Entrenar con TODOS los datos ---
X_train = df[FEATURES].values
y_train = df[OBJETIVO].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

modelo = Ridge(alpha=1.0)
modelo.fit(X_train_scaled, y_train)

# --- Predecir el siguiente mes ---
ultima_fila = df.iloc[-1]
ultima_fecha = ultima_fila["fecha"]
fecha_prediccion = ultima_fecha + pd.DateOffset(months=3)

X_nuevo = ultima_fila[FEATURES].values.reshape(1, -1)
X_nuevo_scaled = scaler.transform(X_nuevo)
prediccion = modelo.predict(X_nuevo_scaled)[0]

# --- Baselines para comparar ---
naive = df.iloc[-1][OBJETIVO]
media_movil = df[OBJETIVO].tail(12).mean()
expect = df.iloc[-1]["expectativas_inflacion"] / 12

print("=" * 50)
print(f"PREDICCION - Inflacion mensual")
print(f"Usando datos hasta: {ultima_fecha.date()}")
print(f"Prediciendo: {fecha_prediccion.date()}")
print("=" * 50)
print(f"  Ridge:          {prediccion:.4f}%")
print(f"  Naive:          {naive:.4f}%")
print(f"  Media movil 12: {media_movil:.4f}%")
print(f"  Expectativas:   {expect:.4f}%")
print("=" * 50)
print(f"\nEl modelo predice que la inflacion de {fecha_prediccion.strftime('%B %Y')}")
print(f"sera aproximadamente {prediccion:.2f}%")