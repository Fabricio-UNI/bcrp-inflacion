# Predicción de la Inflación Mensual del Perú

Pipeline de datos y modelo predictivo para anticipar la variación mensual del IPC de Lima Metropolitana, usando series públicas del Banco Central de Reserva del Perú (BCRP).

## Problemática

El BCRP fija su tasa de referencia mirando hacia adelante: la política monetaria tarda meses en surtir efecto. Anticipar la inflación con 3 meses de ventana tiene valor directo para esa decisión.

**Pregunta:** ¿Se puede predecir la variación mensual del IPC con 3 meses de anticipación usando únicamente información disponible al momento de la predicción?

## Resultados

| Modelo | RMSE | MAE | Ratio vs Naive |
|--------|------|-----|----------------|
| Naive (baseline) | 0.4823 | 0.3583 | 1.000 |
| Media Móvil 12m | 0.3605 | 0.2592 | 0.747 |
| Expectativas BCRP | 0.3500 | 0.2443 | 0.726 |
| **Ridge** | **0.3406** | **0.2457** | **0.706** |

Ridge supera al baseline naive en **29.4%** y a las expectativas de analistas del BCRP en **2.7%**, evaluado sobre 208 predicciones mensuales (2009–2026) con validación walk-forward.

### Variables más relevantes

El tipo de cambio rezagado 5–6 meses domina la predicción — evidencia del traspaso cambiario a precios, consistente con la literatura del BCRP sobre economías pequeñas y abiertas.

### Predicción actual

**Octubre 2026: 0.59%** (variación mensual estimada, verificable con la publicación del BCRP).

## Datos

Tres series mensuales de la API pública de [BCRPData](https://estadisticas.bcrp.gob.pe/estadisticas/series/api) (2003–2026):

- **IPC Lima Metropolitana** — índice base Dic.2021=100 (`PN38705PM`)
- **Tipo de cambio** — dólar, promedio mensual (`PN01234PM`)
- **Expectativas de inflación** — encuesta a 12 meses (`PD12912AM`)

La inflación mensual se calcula como la variación porcentual del IPC.

## Metodología

### Ingeniería de features
Se construyeron 17 variables rezagadas a partir de las 3 series originales:
- Rezagos de inflación, tipo de cambio y expectativas (3, 4, 5, 6 y 12 meses)
- Media móvil de inflación (12 meses)
- Mes del año (estacionalidad)

**Restricción:** ninguna feature usa información posterior al horizonte de predicción (3 meses). Rezago mínimo = horizonte.

### Validación walk-forward
El modelo se evalúa cronológicamente: entrena con datos hasta el mes *t*, predice *t+3*, avanza un mes y repite. Nunca se usa información del futuro para predecir el pasado. Un `train_test_split` aleatorio invalidaría el experimento en series de tiempo.

- Entrenamiento mínimo: 60 meses (5 años)
- 208 predicciones evaluadas
- 3 baselines: naive, media móvil 12 meses, expectativas BCRP

### Modelo
**Ridge Regression** (regresión lineal regularizada, α=1.0). Se eligió por estabilidad con pocos datos (268 observaciones) e interpretabilidad. Con ~280 datos mensuales, modelos más complejos como XGBoost presentan alto riesgo de sobreajuste.

## Estructura del proyecto

```
bcrp-inflacion/
├── app.py                  # Dashboard Streamlit
├── requirements.txt        # Dependencias
├── src/
│   ├── __init__.py
│   ├── config.py           # Códigos de series BCRP
│   ├── extract.py          # Extracción desde la API
│   ├── transform.py        # Limpieza y cálculo de inflación
│   ├── features.py         # Features rezagadas
│   ├── models.py           # Entrenamiento y backtesting
│   ├── visualize.py        # Gráficos matplotlib
│   └── predict.py          # Predicción futura
├── data/
│   ├── raw/                # JSON crudo del BCRP
│   └── processed/          # CSV limpios y resultados
└── .streamlit/
    └── config.toml         # Tema del dashboard
```

## Cómo ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Pipeline completo
python -m src.extract
python -m src.transform
python -m src.features
python -m src.models
python -m src.predict
python -m src.visualize

# Dashboard
streamlit run app.py
```

## Tecnologías

Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, matplotlib, API REST (BCRPData)

## Autor

**Alvaro Fabricio Nuñez Rivas**
Ingeniería de Sistemas — Universidad Nacional de Ingeniería (UNI), Lima, Perú
