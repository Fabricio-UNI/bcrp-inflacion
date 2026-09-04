"""
Paso 5: Entrenar modelos y evaluar con backtesting walk-forward.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# --- Cargar features ---
df = pd.read_csv("data/processed/features.csv", parse_dates=["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

print(f"Datos cargados: {len(df)} filas")
print(f"Rango: {df['fecha'].min().date()} a {df['fecha'].max().date()}")

# --- Definir columnas ---
OBJETIVO = "inflacion_mensual"
EXCLUIR = ["fecha", "inflacion_mensual", "ipc_indice", "tipo_cambio", "expectativas_inflacion"]
FEATURES = [c for c in df.columns if c not in EXCLUIR]

print(f"Features: {len(FEATURES)}")
print(f"Objetivo: {OBJETIVO}")

# --- Configuracion del backtesting ---
# Entrenar con al menos 60 meses (5 anios), predecir el resto uno a uno
MIN_TRAIN = 60
HORIZONTE = 3

print(f"\nBacktesting walk-forward:")
print(f"  Minimo de entrenamiento: {MIN_TRAIN} meses")
print(f"  Ventana de evaluacion: {len(df) - MIN_TRAIN} predicciones")

# --- Almacenar predicciones ---
resultados = {
    "fecha": [],
    "real": [],
    "naive": [],
    "media_movil_12": [],
    "expectativas": [],
    "ridge": [],
}

# --- Walk-forward ---
for i in range(MIN_TRAIN, len(df)):
    # Datos hasta el momento (solo lo que conocerias)
    train = df.iloc[:i]
    test_row = df.iloc[i]

    fecha_pred = test_row["fecha"]
    real = test_row[OBJETIVO]

    # --- Baseline 1: Naive (ultimo valor conocido, rezagado al horizonte) ---
    if i - HORIZONTE >= 0:
        naive = df.iloc[i - HORIZONTE][OBJETIVO]
    else:
        naive = np.nan

    # --- Baseline 2: Media movil 12 meses ---
    if i - HORIZONTE >= 11:
        ventana = df.iloc[i - HORIZONTE - 11: i - HORIZONTE + 1][OBJETIVO]
        media_movil = ventana.mean()
    else:
        media_movil = np.nan

    # --- Baseline 3: Expectativas del BCRP ---
    # Las expectativas son a 12 meses, usamos la de hace HORIZONTE meses
    # como proxy de lo que se esperaba
    if i - HORIZONTE >= 0:
        expect = df.iloc[i - HORIZONTE]["expectativas_inflacion"]
        # Convertir de anual a mensual aproximado
        expect_mensual = expect / 12
    else:
        expect_mensual = np.nan

    # --- Modelo Ridge ---
    X_train = train[FEATURES].values
    y_train = train[OBJETIVO].values
    X_test = test_row[FEATURES].values.reshape(1, -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    modelo = Ridge(alpha=1.0)
    modelo.fit(X_train_scaled, y_train)
    pred_ridge = modelo.predict(X_test_scaled)[0]

    # --- Guardar ---
    resultados["fecha"].append(fecha_pred)
    resultados["real"].append(real)
    resultados["naive"].append(naive)
    resultados["media_movil_12"].append(media_movil)
    resultados["expectativas"].append(expect_mensual)
    resultados["ridge"].append(pred_ridge)

# --- Crear DataFrame de resultados ---
df_res = pd.DataFrame(resultados)
df_res = df_res.dropna()

print(f"\nPredicciones evaluadas: {len(df_res)}")
print(f"Periodo: {df_res['fecha'].min().date()} a {df_res['fecha'].max().date()}")

# --- Calcular metricas ---
print("\n" + "=" * 60)
print("RESULTADOS - Horizonte 3 meses")
print("=" * 60)

modelos = ["naive", "media_movil_12", "expectativas", "ridge"]

metricas = []
for nombre in modelos:
    rmse = np.sqrt(mean_squared_error(df_res["real"], df_res[nombre]))
    mae = mean_absolute_error(df_res["real"], df_res[nombre])
    metricas.append({"modelo": nombre, "RMSE": round(rmse, 4), "MAE": round(mae, 4)})

df_metricas = pd.DataFrame(metricas)

# Calcular ratio contra naive
rmse_naive = df_metricas.loc[df_metricas["modelo"] == "naive", "RMSE"].values[0]
df_metricas["ratio_vs_naive"] = round(df_metricas["RMSE"] / rmse_naive, 3)

print(df_metricas.to_string(index=False))

print("\nInterpretacion del ratio:")
print("  < 1.0 = mejor que el naive (gana)")
print("  = 1.0 = igual que el naive (empata)")
print("  > 1.0 = peor que el naive (pierde)")

# --- Features mas importantes del Ridge (ultimo modelo entrenado) ---
importancias = pd.Series(modelo.coef_, index=FEATURES).abs().sort_values(ascending=False)
print("\nTop 5 features del Ridge (por coeficiente absoluto):")
print(importancias.head().to_string())

# --- Guardar resultados ---
df_res.to_csv("data/processed/predicciones.csv", index=False)
df_metricas.to_csv("data/processed/metricas.csv", index=False)
print("\nGuardado en data/processed/predicciones.csv y metricas.csv")