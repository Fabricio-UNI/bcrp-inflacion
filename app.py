import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Inflación Perú — BCRP", page_icon="📉", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* FORZAR FONDO CREMA */
    .stApp { background-color: #FDF5E6 !important; }
    header[data-testid="stHeader"] { background-color: #FDF5E6 !important; }
    section[data-testid="stSidebar"] { background-color: #FAE8C8 !important; }
    div[data-testid="stExpander"] { background-color: #FAE8C8 !important; border-radius: 12px; }
    div[data-testid="stDataFrame"] { background-color: #FFFFFF !important; border-radius: 8px; }
    
    /* FORZAR TEXTO OSCURO */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div { color: #1a1a2e !important; }
    .stMarkdown, .stMarkdown p { color: #1a1a2e !important; }
    
    .block-container { padding: 2rem 3rem; max-width: 1300px; }
    h1 { font-weight: 700 !important; font-size: 2.2rem !important; color: #1a1a2e !important; letter-spacing: -0.5px; }
    h2 { 
        font-weight: 600 !important; 
        font-size: 1.3rem !important; 
        color: #16213e !important; 
        margin-top: 2rem; 
        margin-bottom: 1.5rem !important; /* Agrega esta línea para separar el título del desplegable */
        padding-bottom: 0.4rem; 
        border-bottom: 2px solid #d4a574 !important; 
    }
    h4 { color: #1a1a2e !important; }
    
    .subtitle { font-size: 1rem; color: #6c757d !important; font-weight: 300; margin-bottom: 2rem; line-height: 1.6; }
    
    .prediction-box { 
        background: linear-gradient(135deg, #e53935 0%, #F5635B 100%) !important; 
        border-radius: 16px; padding: 2rem; margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(229,57,53,0.3);
    }

    .prediction-box h3 { color: #f0d0a0 !important; font-weight: 300; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.3rem; }
    .prediction-value { font-size: 3rem; font-weight: 700; color: #ffffff !important; margin: 0.3rem 0; }
    .prediction-detail { color: #f0d0a0 !important; font-size: 0.85rem; }
    
    .insight-box { 
        background: #f5ead6 !important; border-left: 3px solid #B22222; border-radius: 0 8px 8px 0; 
        padding: 1rem 1.5rem; margin: 1rem 0; font-size: 0.9rem; color: #374151 !important; line-height: 1.6; 
    }
    
    .method-box { 
        background: #FFF8EE !important; border: 1px solid #d4a574; border-radius: 12px; 
        padding: 1.5rem; margin: 0.5rem 0; font-size: 0.9rem; color: #374151 !important; line-height: 1.7; 
    }
    .method-box h4 { color: #8B1A1A !important; font-size: 1rem; margin-bottom: 0.5rem; }
    
    .metric-card { 
        background: #FFF8EE !important; border: 1px solid #d4a574; border-radius: 12px; 
        padding: 1.2rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e !important; margin: 0.2rem 0; }
    .metric-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; color: #8B6914 !important; font-weight: 600; }
    .metric-sub { font-size: 0.8rem; margin-top: 0.2rem; }
    .winner { color: #0d7a6e !important; font-weight: 600; }
    .loser { color: #B22222 !important; font-weight: 600; }
    
    .footer { 
        margin-top: 3rem; padding-top: 1.5rem; border-top: 2px solid #d4a574; 
        color: #8B6914 !important; font-size: 0.8rem; text-align: center; 
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Multiselect y widgets */
    div[data-baseweb="select"] { background-color: #FFF8EE !important; }
    div[data-baseweb="tag"] { background-color: #B22222 !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="Inter", size=12, color="#1a1a2e"),
)

@st.cache_data
def cargar_datos():
    pred = pd.read_csv("data/processed/predicciones.csv", parse_dates=["fecha"])
    metricas = pd.read_csv("data/processed/metricas.csv")
    features = pd.read_csv("data/processed/features.csv", parse_dates=["fecha"])
    return pred, metricas, features

df_pred, df_metricas, df_features = cargar_datos()

OBJETIVO = "inflacion_mensual"
EXCLUIR = ["fecha", "inflacion_mensual", "ipc_indice", "tipo_cambio", "expectativas_inflacion"]
FEATURES = [c for c in df_features.columns if c not in EXCLUIR]
X = df_features[FEATURES].values
y = df_features[OBJETIVO].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
modelo = Ridge(alpha=1.0)
modelo.fit(X_scaled, y)

# ============================================================
# HEADER
# ============================================================
st.markdown("# 📉 Predicción de la Inflación Mensual del Perú")
st.markdown('<p class="subtitle">Pipeline de datos y modelo predictivo sobre series del Banco Central de Reserva del Perú (BCRP)<br>Horizonte: 3 meses · Validación: Walk-forward · Periodo: 2003–2026</p>', unsafe_allow_html=True)

# ============================================================
# PREDICCION DESTACADA
# ============================================================
ultima = df_features.iloc[-1]
X_nuevo = ultima[FEATURES].values.reshape(1, -1)
X_nuevo_scaled = scaler.transform(X_nuevo)
pred_octubre = modelo.predict(X_nuevo_scaled)[0]
naive_val = df_features.iloc[-1][OBJETIVO]
media_val = df_features[OBJETIVO].tail(12).mean()
expect_val = df_features.iloc[-1]["expectativas_inflacion"] / 12

col_pred, col_comp = st.columns([1, 2])
with col_pred:
    st.markdown(f'<div class="prediction-box"><h3>Predicción · Octubre 2026</h3><div class="prediction-value">{pred_octubre:.2f}%</div><div class="prediction-detail">Variación mensual estimada del IPC<br>Datos hasta julio 2026<br>Verificable con publicación del BCRP</div></div>', unsafe_allow_html=True)
with col_comp:
    st.markdown("**Comparación de predicciones para octubre 2026**")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=["Ridge", "Expectativas BCRP", "Media Móvil 12m", "Naive"], y=[pred_octubre, expect_val, media_val, naive_val], marker_color=["#BE4848", "#0d7a6e", "#d4a574", "#8B6914"], text=[f"{v:.3f}%" for v in [pred_octubre, expect_val, media_val, naive_val]], textposition="outside", textfont=dict(size=13, family="Inter", color="#1a1a2e"), hovertemplate="%{x}: %{y:.4f}%<extra></extra>", showlegend=False))
    fig_comp.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(title="Variación % mensual", gridcolor="#e8dcc8", title_font=dict(size=11, color="#6c757d")), **PLOTLY_LAYOUT)
    st.plotly_chart(fig_comp, use_container_width=True)

# ============================================================
# RESULTADOS
# ============================================================
st.markdown("## Resultados del modelo")
ridge_row = df_metricas[df_metricas["modelo"] == "ridge"].iloc[0]
mejora = (1 - ridge_row["ratio_vs_naive"]) * 100
st.markdown(f'<div class="insight-box">El modelo Ridge supera al baseline naive en <strong>{mejora:.1f}%</strong>, evaluado sobre <strong>208 predicciones mensuales</strong> (abril 2009 – julio 2026) con validación walk-forward. También supera a las expectativas de analistas encuestados por el BCRP, lo cual es un resultado no trivial dado que esas expectativas incorporan juicio humano experto.</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
modelos_info = {"ridge": ("Ridge", "#B22222"), "expectativas": ("Expectativas", "#0d7a6e"), "media_movil_12": ("Media Móvil 12m", "#d4a574"), "naive": ("Naive (baseline)", "#8B6914")}
for i, (key, (nombre, color)) in enumerate(modelos_info.items()):
    row = df_metricas[df_metricas["modelo"] == key].iloc[0]
    col = [col1, col2, col3, col4][i]
    ratio = row["ratio_vs_naive"]
    if ratio < 1:
        delta_text = f"↓ {(1-ratio)*100:.1f}% vs naive"
        cls = "winner"
    elif ratio == 1:
        delta_text = "— baseline"
        cls = ""
    else:
        delta_text = f"↑ {(ratio-1)*100:.1f}% vs naive"
        cls = "loser"
    col.markdown(f'<div class="metric-card"><div class="metric-label">{nombre}</div><div class="metric-value">{row["RMSE"]:.3f}</div><div style="font-size:0.75rem; color:#8B6914;">RMSE · MAE: {row["MAE"]:.3f}</div><div class="metric-sub {cls}">{delta_text}</div></div>', unsafe_allow_html=True)

# ============================================================
# ANALISIS VISUAL
# ============================================================
st.markdown("## Análisis visual")
st.markdown("#### Inflación real vs predicciones")
modelos_mostrar = st.multiselect("Selecciona modelos para comparar:", ["ridge", "naive", "media_movil_12", "expectativas"], default=["ridge", "naive"], format_func=lambda x: {"ridge": "Ridge", "naive": "Naive", "media_movil_12": "Media Móvil 12m", "expectativas": "Expectativas BCRP"}[x])

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_pred["fecha"], y=df_pred["real"], name="Real", line=dict(color="#1a1a2e", width=2), hovertemplate="%{x|%b %Y}: %{y:.3f}%<extra>Real</extra>"))
colores_linea = {"ridge": "#B22222", "naive": "#8B6914", "media_movil_12": "#d4a574", "expectativas": "#0d7a6e"}
estilos = {"ridge": "solid", "naive": "dot", "media_movil_12": "dash", "expectativas": "dashdot"}
nombres_linea = {"ridge": "Ridge", "naive": "Naive", "media_movil_12": "Media Móvil 12m", "expectativas": "Expectativas"}
for m in modelos_mostrar:
    fig.add_trace(go.Scatter(x=df_pred["fecha"], y=df_pred[m], name=nombres_linea[m], line=dict(color=colores_linea[m], width=1.5, dash=estilos[m]), hovertemplate=f"%{{x|%b %Y}}: %{{y:.3f}}%<extra>{nombres_linea[m]}</extra>"))
fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis=dict(title="Variación % mensual del IPC", gridcolor="#e8dcc8", zeroline=True, zerolinecolor="#d4a574", title_font=dict(size=11, color="#6c757d")), xaxis=dict(gridcolor="#e8dcc8"), hovermode="x unified", **PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("#### Predicción vs Real")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_pred["real"], y=df_pred["ridge"], mode="markers", marker=dict(color="#B22222", size=6, opacity=0.5, line=dict(width=0.5, color="white")), name="Predicciones", hovertemplate="Real: %{x:.3f}%<br>Ridge: %{y:.3f}%<extra></extra>"))
    min_v = min(df_pred["real"].min(), df_pred["ridge"].min()) - 0.2
    max_v = max(df_pred["real"].max(), df_pred["ridge"].max()) + 0.2
    fig2.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(color="#8B6914", dash="dash", width=1), name="Predicción perfecta"))
    fig2.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(title="Inflación real (%)", gridcolor="#e8dcc8", title_font=dict(size=10, color="#6c757d")), yaxis=dict(title="Predicción Ridge (%)", gridcolor="#e8dcc8", title_font=dict(size=10, color="#6c757d")), legend=dict(font=dict(size=9), yanchor="top", y=0.99, xanchor="left", x=0.01), **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown("#### Distribución del error")
    errores = df_pred["ridge"] - df_pred["real"]
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=errores, nbinsx=25, marker_color="#B22222", opacity=0.7, hovertemplate="Error: %{x:.3f}%<br>Frecuencia: %{y}<extra></extra>"))
    fig3.add_vline(x=0, line_dash="dash", line_color="#8B6914", line_width=1)
    fig3.add_vline(x=errores.mean(), line_dash="dot", line_color="#0d7a6e", line_width=1.5, annotation_text=f"media: {errores.mean():.3f}", annotation_position="top right", annotation_font=dict(size=10, color="#0d7a6e"))
    fig3.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(title="Error (predicción − real)", gridcolor="#e8dcc8", title_font=dict(size=10, color="#6c757d")), yaxis=dict(title="Frecuencia", gridcolor="#e8dcc8", title_font=dict(size=10, color="#6c757d")), showlegend=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

with col_c:
    st.markdown("#### Variables más relevantes")
    importancias = pd.DataFrame({"Feature": FEATURES, "Coeficiente": modelo.coef_, "Importancia": np.abs(modelo.coef_)}).sort_values("Importancia", ascending=True).tail(10)
    colors_feat = ["#0d7a6e" if c > 0 else "#B22222" for c in importancias["Coeficiente"]]
    labels = importancias["Feature"].str.replace("_", " ").str.replace("inflacion", "infl.").str.replace("expect", "exp.")
    fig4 = go.Figure(go.Bar(x=importancias["Importancia"], y=labels, orientation="h", marker_color=colors_feat, hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig4.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(title="Coeficiente absoluto", gridcolor="#e8dcc8", title_font=dict(size=10, color="#6c757d")), yaxis=dict(title="", tickfont=dict(size=10)), **PLOTLY_LAYOUT)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown('<div class="insight-box"><strong>Lectura:</strong> Los coeficientes en <span style="color:#0d7a6e">verde</span> tienen efecto positivo y los <span style="color:#B22222">rojos</span> negativo. El tipo de cambio rezagado 5–6 meses domina la predicción, evidenciando el traspaso cambiario: cuando el dólar sube, los precios de importaciones tardan medio año en impactar el IPC. Las expectativas de inflación y el patrón anual (lag 12) complementan el modelo. La distribución de errores muestra un sesgo cercano a cero, lo que indica que el modelo no sobreestima ni subestima sistemáticamente.</div>', unsafe_allow_html=True)

# ============================================================
# METODOLOGIA
# ============================================================
st.markdown("## Metodología")
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown('<div class="method-box"><h4>📥 1. Extracción de datos</h4>Tres series mensuales de la API pública de BCRPData (2003–2026):<ul style="margin-top:0.5rem;"><li><strong>IPC</strong> Lima Metropolitana (índice)</li><li><strong>Tipo de cambio</strong> dólar (promedio mensual)</li><li><strong>Expectativas de inflación</strong> a 12 meses (encuesta)</li></ul>La inflación mensual se calcula como variación porcentual del IPC.</div>', unsafe_allow_html=True)
with col_m2:
    st.markdown(f'<div class="method-box"><h4>🔧 2. Ingeniería de features</h4>Se construyeron <strong>{len(FEATURES)} variables</strong> rezagadas a partir de las 3 series originales:<ul style="margin-top:0.5rem;"><li><strong>Rezagos</strong> de inflación, tipo de cambio y expectativas (3, 4, 5, 6 y 12 meses)</li><li><strong>Media móvil</strong> de inflación (12 meses)</li><li><strong>Mes del año</strong> (estacionalidad)</li></ul>Ningún feature usa información posterior al horizonte de predicción. Rezago mínimo = horizonte.</div>', unsafe_allow_html=True)
with col_m3:
    st.markdown('<div class="method-box"><h4>📊 3. Validación walk-forward</h4>Se evalúa el modelo de forma cronológica, sin mezclar pasado y futuro:<ul style="margin-top:0.5rem;"><li>Entrenamiento mínimo: <strong>60 meses</strong></li><li>En cada paso se reentrena con un mes más</li><li>Se compara contra 3 baselines</li><li><strong>208 predicciones</strong> evaluadas (2009–2026)</li></ul>No se usa train/test split aleatorio — eso invalidaría el experimento en series de tiempo.</div>', unsafe_allow_html=True)

st.markdown('<div class="method-box" style="margin-top:0.5rem;"><h4>🧠 Sobre el modelo</h4><strong>Ridge Regression</strong> es una regresión lineal regularizada. Busca los pesos óptimos para combinar las 17 features en una predicción, con una penalización (α=1.0) que evita pesos extremos y reduce el sobreajuste. Se eligió por su estabilidad con pocos datos (268 observaciones) y su interpretabilidad: los coeficientes indican directamente la dirección y magnitud del efecto de cada variable.<br><br><strong>¿Por qué no modelos más complejos?</strong> Con ~280 observaciones mensuales el riesgo de sobreajuste supera el beneficio de capturar no linealidades. Ridge ofrece la mejor relación entre complejidad y rendimiento para este volumen de datos.</div>', unsafe_allow_html=True)

# ============================================================
# DATOS DETALLADOS
# ============================================================
st.markdown("## Datos detallados")
with st.expander("📋 Ver predicciones mes a mes"):
    df_show = df_pred.copy()
    df_show["fecha"] = df_show["fecha"].dt.strftime("%Y-%m")
    df_show = df_show.rename(columns={"fecha": "Mes", "real": "Real", "naive": "Naive", "media_movil_12": "Media Móvil", "expectativas": "Expectativas", "ridge": "Ridge"})
    st.dataframe(df_show, use_container_width=True, height=400)
with st.expander("📊 Ver tabla de métricas completa"):
    st.dataframe(df_metricas, use_container_width=True)
with st.expander("🔧 Ver features utilizadas"):
    st.write(f"**Total:** {len(FEATURES)} features")
    for feat in FEATURES:
        if "inflacion_lag" in feat: st.write(f"- `{feat}`: Inflación mensual rezagada")
        elif "tc_lag" in feat: st.write(f"- `{feat}`: Tipo de cambio dólar rezagado")
        elif "expect_lag" in feat: st.write(f"- `{feat}`: Expectativas de inflación rezagadas")
        elif feat == "inflacion_ma12": st.write(f"- `{feat}`: Media móvil de inflación (12 meses)")
        elif feat == "mes": st.write(f"- `{feat}`: Mes del año (1–12, estacionalidad)")

# ============================================================
# FOOTER
# ============================================================
st.markdown('<div class="footer"><strong>Fuente:</strong> BCRPData API — Banco Central de Reserva del Perú&nbsp;&nbsp;·&nbsp;&nbsp;<strong>Modelo:</strong> Ridge Regression&nbsp;&nbsp;·&nbsp;&nbsp;<strong>Validación:</strong> Walk-forward (208 meses)<br><strong>Autor:</strong> Alvaro Fabricio Nuñez Rivas&nbsp;&nbsp;·&nbsp;&nbsp;Ingeniería de Sistemas, Universidad Nacional de Ingeniería<br><br><span style="font-size: 0.7rem; color: #b8a080;">Proyecto académico. Las predicciones no constituyen asesoría financiera ni representan la posición del BCRP.</span></div>', unsafe_allow_html=True)