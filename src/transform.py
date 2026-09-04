"""
Paso 2: Limpiar datos crudos del BCRP y construir el dataset base.
"""
import json
import pandas as pd

# --- Leer el JSON crudo ---
with open("data/raw/bcrp_mensual.json", "r", encoding="utf-8") as f:
    datos = json.load(f)

# --- Extraer nombres de series ---
nombres_series = [s["name"] for s in datos["config"]["series"]]
print("Series encontradas:")
for i, nombre in enumerate(nombres_series):
    print(f"  [{i}] {nombre}")

# --- Construir el DataFrame ---
filas = []
for periodo in datos["periods"]:
    fila = {"periodo": periodo["name"]}
    for i, valor in enumerate(periodo["values"]):
        fila[f"serie_{i}"] = valor
    filas.append(fila)

df = pd.DataFrame(filas)

# --- Renombrar columnas ---
df.columns = ["periodo", "expectativas_inflacion", "tipo_cambio", "ipc_indice"]

# --- Convertir periodo a fecha ---
meses = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Ago": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12
}

def parsear_periodo(texto):
    partes = texto.split(".")
    mes = meses[partes[0]]
    anio = int(partes[1])
    return pd.Timestamp(year=anio, month=mes, day=1)

df["fecha"] = df["periodo"].apply(parsear_periodo)

# --- Convertir valores a numeros ---
for col in ["ipc_indice", "tipo_cambio", "expectativas_inflacion"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Calcular inflacion mensual (variacion % del IPC) ---
df = df.sort_values("fecha").reset_index(drop=True)
df["inflacion_mensual"] = df["ipc_indice"].pct_change() * 100

# --- Revisar datos faltantes ---
print("\nDatos faltantes por columna:")
print(df.isnull().sum())

print(f"\nRango: {df['fecha'].min()} a {df['fecha'].max()}")
print(f"Filas totales: {len(df)}")
print(f"Filas con inflacion calculada: {df['inflacion_mensual'].notna().sum()}")

# --- Guardar ---
df_limpio = df[["fecha", "inflacion_mensual", "tipo_cambio", "expectativas_inflacion"]].dropna()
df_limpio.to_csv("data/processed/datos_limpios.csv", index=False)

print(f"\nGuardado en data/processed/datos_limpios.csv ({len(df_limpio)} filas)")
print("\nPrimeras 5 filas:")
print(df_limpio.head().to_string(index=False))
print("\nUltimas 5 filas:")
print(df_limpio.tail().to_string(index=False))