import streamlit as st
import numpy as np
import scipy.stats as stats
import xgboost as xgb
import json
import pandas as pd

# ==============================================================================
# 🎨 CONFIGURACIÓN DE LA INTERFAZ DE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Football Predictor AI", layout="wide", page_icon="📊")

st.title("⚽ Ecosistema Predictivo: Goles, Córners y Tarjetas")
st.markdown("---")

# 1. CARGA DE ACTIVOS MÁSTER (Cacheada para optimizar velocidad)
@st.cache_data
def cargar_base_datos():
    with open("bayes_data_6years.json", "r") as f:
        return json.load(f)

@st.cache_resource
def cargar_modelo_xgb():
    modelo = xgb.XGBClassifier()
    modelo.load_model("xgboost_goles_model.json")
    return modelo

try:
    db = cargar_base_datos()
    modelo_xgb = cargar_modelo_xgb()
except Exception as e:
    st.error(f"❌ Error al cargar los archivos base: {e}. Asegúrate de que estén en la misma carpeta.")
    st.stop()

# ==============================================================================
# 🎮 INTERFAZ DE SELECCIÓN (CONTROLES)
# ==============================================================================
lista_equipos = sorted(list(db["teams"].keys()))

col1, col2 = st.columns(2)

with col1:
    equipo_local = st.selectbox("🏟️ Selecciona el Equipo Local:", lista_equipos, index=lista_equipos.index("Arsenal") if "Arsenal" in lista_equipos else 0)

with col2:
    equipo_visitante = st.selectbox("✈️ Selecciona el Equipo Visitante:", lista_equipos, index=lista_equipos.index("Man City") if "Man City" in lista_equipos else 1)

if equipo_local == equipo_visitante:
    st.warning("⚠️ El equipo local y el visitante no pueden ser el mismo.")
    st.stop()

# ==============================================================================
# 🧮 CÁLCULOS MATEMÁTICOS DEL SIMULADOR
# ==============================================================================
data_l = db["teams"][equipo_local]
data_v = db["teams"][equipo_visitante]
intercept = db["intercept"]
home_effect = db["home_effect"]

# A) Mercado de Goles (Poisson)
lambda_goles_l = np.exp(intercept + home_effect + data_l["attack"] - data_v["defense"])
lambda_goles_v = np.exp(intercept - home_effect + data_v["attack"] - data_l["defense"])
xG_total = lambda_goles_l + lambda_goles_v

# B) Auditoría XGBoost V2 (Probabilidad de Over 2.5)
vector_partido = pd.DataFrame([{
    'home_effect_global': home_effect,
    'idx_ataque_local': data_l["attack"],
    'idx_defensa_local': data_l["defense"],
    'idx_ataque_visita': data_v["attack"],
    'idx_defensa_visita': data_v["defense"]
}])
prob_xgb = modelo_xgb.predict_proba(vector_partido)[0][1] * 100

# C) Córners (Poisson)
lambda_corners_l = (data_l["corners_att"] + data_v["corners_def"]) / 2
lambda_corners_v = (data_v["corners_att"] + data_l["corners_def"]) / 2
total_corners = lambda_corners_l + lambda_corners_v
prob_over_95_c = (1 - stats.poisson.cdf(9, total_corners)) * 100

# D) Tarjetas (Poisson)
total_tarjetas = data_l["cards_recv"] + data_v["cards_recv"]
prob_over_35_t = (1 - stats.poisson.cdf(3, total_tarjetas)) * 100

# ==============================================================================
# 📊 PRESENTACIÓN VISUAL EN PANTALLA
# ==============================================================================
st.markdown(f"### Proyecciones Avanzadas: **{equipo_local.upper()}** vs **{equipo_visitante.upper()}**")

# Estructura de pestañas para dividir los mercados limpiamente
tab_goles, tab_corners, tab_tarjetas = st.tabs(["⚽ Goles (XGBoost + Poisson)", "📐 Córners (Poisson)", "🟨 Tarjetas (6 Años)"])

with tab_goles:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"xG {equipo_local}", f"{lambda_goles_l:.2f}")
    c2.metric(f"xG {equipo_visitante}", f"{lambda_goles_v:.2f}")
    c3.metric("Línea Esperada (Total)", f"{xG_total:.2f}")
    
    st.markdown("#### 🤖 Auditoría de Machine Learning:")
    if prob_xgb >= 55.0:
        st.success(f"🔥 **Señal de Valor Detectada (+EV):** Probabilidad IA de **Over 2.5 Goles** es del **{prob_xgb:.1f}%**")
    else:
        st.info(f"📊 Probabilidad IA de **Over 2.5 Goles**: **{prob_xgb:.1f}%** (Sin ventaja clara sobre el mercado)")

with tab_corners:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Córners {equipo_local}", f"{lambda_corners_l:.2f}")
    c2.metric(f"Córners {equipo_visitante}", f"{lambda_corners_v:.2f}")
    c3.metric("Total Córners Previstos", f"{total_corners:.2f}")
    
    st.write(f"🎯 Probabilidad de **Más de 9.5 Córners Totales**: **{prob_over_95_c:.1f}%**")

with tab_tarjetas:
    c1, c2 = st.columns(2)
    c1.metric("Índice de Tarjetas Esperadas", f"{total_tarjetas:.2f}")
    c2.write(f"🎯 Probabilidad de **Más de 3.5 Tarjetas Totales**: **{prob_over_35_t:.1f}%**")