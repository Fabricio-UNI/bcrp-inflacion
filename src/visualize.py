"""
Paso 6: Visualizaciones del proyecto.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Cargar datos ---
df_pred = pd.read_csv("data/processed/predicciones.csv", parse_dates=["fecha"])
df_metricas = pd.read_csv("data/processed/metricas.csv")

print(f"Predicciones cargadas: {len(df_pred)} meses")

# --- Grafico 1: Linea temporal ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_pred["fecha"], df_pred["real"], label="Real", color="black", linewidth=1.5)
ax.plot(df_pred["fecha"], df_pred["ridge"], label="Ridge", color="blue", linewidth=1, alpha=0.8)
ax.plot(df_pred["fecha"], df_pred["naive"], label="Naive", color="red", linewidth=0.8, alpha=0.5, linestyle="--")
ax.set_title("Inflación mensual: Real vs Predicciones (horizonte 3 meses)", fontsize=13)
ax.set_ylabel("Variación % mensual del IPC")
ax.set_xlabel("Fecha")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("data/processed/grafico_linea_temporal.png", dpi=150)
print("Guardado: grafico_linea_temporal.png")
plt.close()

# --- Grafico 2: Barras de error ---
fig, ax = plt.subplots(figsize=(8, 5))
colores = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
barras = ax.bar(df_metricas["modelo"], df_metricas["RMSE"], color=colores)
ax.set_title("RMSE por modelo (menor es mejor)", fontsize=13)
ax.set_ylabel("RMSE")
for barra, ratio in zip(barras, df_metricas["ratio_vs_naive"]):
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.005,
            f"ratio: {ratio}", ha="center", fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("data/processed/grafico_barras_error.png", dpi=150)
print("Guardado: grafico_barras_error.png")
plt.close()

# --- Grafico 3: Dispersion real vs prediccion ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df_pred["real"], df_pred["ridge"], alpha=0.5, color="blue", s=20)
min_val = min(df_pred["real"].min(), df_pred["ridge"].min()) - 0.2
max_val = max(df_pred["real"].max(), df_pred["ridge"].max()) + 0.2
ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Prediccion perfecta")
ax.set_title("Ridge: Predicción vs Real", fontsize=13)
ax.set_xlabel("Inflación real (%)")
ax.set_ylabel("Predicción Ridge (%)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("data/processed/grafico_dispersion.png", dpi=150)
print("Guardado: grafico_dispersion.png")
plt.close()

print("\nTodos los graficos guardados en data/processed/")