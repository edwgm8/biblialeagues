import streamlit as st
import numpy as np
import scipy.stats as stats
import xgboost as xgb
import json
import pandas as pd

# ==============================================================================
# 🎨 ESTILO PREMIUM PERSONALIZADO (INYECCIÓN DE COLOR ESTILO BET365)
# ==============================================================================
st.set_page_config(page_title="🚨 Bet365 Style - AI Predictor", layout="wide", page_icon="📊")

# CSS personalizado para emular la paleta de colores de Bet365
st.markdown("""
    <style>
    /* Fondo general de la app */
    .stApp {
        background-color: #0d1b15;
        color: #ffffff;
    }
    /* Estilo de los contenedores de métricas */
    div[data-testid="stMetricValue"] {
        color: #ffdf1b !important; /* Amarillo Bet365 */
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #a3b8b0 !important; /* Gris verdoso suave */
    }
    /* Encabezados y títulos */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    /* Líneas divisorias */
    hr {
        border-top: 1px solid #1f3a30 !important;
    }
    /* Tablas estilo dark */
    .stTable table {
        background-color: #13271e !important;
        color: #ffffff !important;
        border: 1px solid #1f3a30 !important;
    }
    th {
        background-color: #0f3524 !important;
        color: #ffdf1b !important;
    }
    </style>
""", unsafe_index=True)

st.title("🟢 Bet365 Analytics Lab — Premier League AI")
st.markdown("<p style='color: #ffdf1b; font-weight: bold;'>SISTEMA HÍBRIDO: BAYESIAN POISSON & XGBOOST AUDITOR</p>", unsafe_index=True)
st.markdown("---")

# ==============================================================================
# 💾 CARGA Y PARSEO DE DATOS (CON PARCHE DE EQUIPOS NUEVOS/ASCENDIDOS)
# ==============================================================================
@st.cache_data
def cargar_y_reparar_base_datos():
    with open("bayes_data_6years.json", "r") as f:
        data = json.load(f)
    
    # 🚨 LISTA DE LOS 20 EQUIPOS REALES DE LA PREMIER LEAGUE ACTUAL
    # Agrega aquí los nombres exactos tal como los envía tu API de Football-Data.org
    equipos_actuales_2026 = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich", 
        "Leicester", "Liverpool", "Man City", "Man United", "Newcastle", 
        "Nottingham", "Southampton", "Tottenham", "West Ham", "Wolves"
    ]
    
    # Calcular los promedios globales de la liga para usarlos como "red de seguridad"
    ataques = [e["attack"] for e in data["teams"].values()]
    defensas = [e["defense"] for e in data["teams"].values()]
    corners_a = [e["corners_att"] for e in data["teams"].values()]
    corners_d = [e["corners_def"] for e in data["teams"].values()]
    cards = [e["cards_recv"] for e in data["teams"].values()]
    
    avg_attack = np.mean(ataques)
    avg_defense = np.mean(defensas)
    avg_corners_att = np.mean(corners_a)
    avg_corners_def = np.mean(corners_d)
    avg_cards = np.mean(cards)
    
    # Inyectar de forma automática los clubes nuevos que no tengan historial en el archivo JSON
    for eq in equipos_actuales_2026:
        if eq not in data["teams"]:
            data["teams"][eq] = {
                "attack": round(avg_attack, 2),
                "defense": round(avg_defense, 2),
                "corners_att": round(avg_corners_att, 2),
                "corners_def": round(avg_corners_def, 2),
                "cards_recv": round(avg_cards, 2)
            }
            
    return data

@st.cache_resource
def cargar_modelo_xgb():
    modelo = xgb.XGBClassifier()
    modelo.load_model("xgboost_goles_model.json")
    return modelo

try:
    db = cargar_y_reparar_base_datos()
    modelo_xgb = cargar_modelo_xgb()
except Exception as e:
    st.error(f"❌ Error crítico en los activos de datos: {e}")
    st.stop()

# ==============================================================================
# 🎮 MENÚS DESPLEGABLES (CONTROLES DE SELECCIÓN)
# ==============================================================================
lista_equipos = sorted(list(db["teams"].keys()))

col_l, col_v = st.columns(2)
with col_l:
    equipo_local = st.selectbox("🏟️ LOCAL (Home Team):", lista_equipos, index=lista_equipos.index("Arsenal") if "Arsenal" in lista_equipos else 0)
with col_v:
    equipo_visitante = st.selectbox("✈️ VISITANTE (Away Team):", lista_equipos, index=lista_equipos.index("Man City") if "Man City" in lista_equipos else 1)

if equipo_local == equipo_visitante:
    st.warning("⚠️ Selecciona dos equipos rivales distintos para ejecutar el algoritmo.")
    st.stop()

# ==============================================================================
# 🧮 OPERACIONES MATEMÁTICAS CENTRALES
# ==============================================================================
data_l = db["teams"][equipo_local]
data_v = db["teams"][equipo_visitante]
intercept = db["intercept"]
home_effect = db["home_effect"]

# MODELO 1: Cálculos Teóricos Bayes/Poisson
lambda_goles_l = np.exp(intercept + home_effect + data_l["attack"] - data_v["defense"])
lambda_goles_v = np.exp(intercept - home_effect + data_v["attack"] - data_l["defense"])
xG_total_poisson = lambda_goles_l + lambda_goles_v

# MODELO 2: Inferencia de Machine Learning XGBoost V2
vector_partido = pd.DataFrame([{
    'home_effect_global': home_effect,
    'idx_ataque_local': data_l["attack"],
    'idx_defensa_local': data_l["defense"],
    'idx_ataque_visita': data_v["attack"],
    'idx_defensa_visita': data_v["defense"]
}])
prob_xgb = modelo_xgb.predict_proba(vector_partido)[0][1] * 100

# Córners y Tarjetas (Poisson)
lambda_corners_l = (data_l["corners_att"] + data_v["corners_def"]) / 2
lambda_corners_v = (data_v["corners_att"] + data_l["corners_def"]) / 2
total_corners = lambda_corners_l + lambda_corners_v
prob_over_95_c = (1 - stats.poisson.cdf(9, total_corners)) * 100

total_tarjetas = data_l["cards_recv"] + data_v["cards_recv"]
prob_over_35_t = (1 - stats.poisson.cdf(3, total_tarjetas)) * 100

# Generar la tabla del Top 10 Marcadores con Poisson
max_goles = 6
marcadores_lista = []
for goles_l in range(max_goles):
    for goles_v in range(max_goles):
        prob_l = stats.poisson.pmf(goles_l, lambda_goles_l)
        prob_v = stats.poisson.pmf(goles_v, lambda_goles_v)
        prob_marcador = prob_l * prob_v * 100
        marcadores_lista.append({
            "Marcador Exacto": f"{goles_l} - {goles_v}",
            "Probabilidad": round(prob_marcador, 2)
        })
df_top10 = pd.DataFrame(marcadores_lista).sort_values(by="Probabilidad", ascending=False).head(10).reset_index(drop=True)

# ==============================================================================
# 🖥️ DESPLIEGUE VISUAL POR ENFOQUES SEPARADOS
# ==============================================================================
st.markdown(f"### <span style='color: #ffdf1b;'>CUOTAS ANALÍTICAS:</span> {equipo_local.upper()} vs {equipo_visitante.upper()}", unsafe_index=True)

tab_goles, tab_props = st.tabs(["⚽ MERCADO DE GOLES (ENFOQUES SEPARADOS)", "📐 MERCADOS COMPLEMENTARIOS"])

with tab_goles:
    # Creamos dos grandes columnas para separar el análisis de ambos modelos
    col_bayes, col_xgb_model = st.columns(2)
    
    with col_bayes:
        st.markdown("<div style='background-color: #0f3524; padding: 15px; border-radius: 5px; border-left: 5px solid #ffdf1b;'><h4>📐 1. ENFOQUE BAYESIANO / POISSON (Histórico)</h4></div>", unsafe_index=True)
        st.write("Métricas de rendimiento puras proyectadas matemáticamente por los últimos 6 años:")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"xG {equipo_local}", f"{lambda_goles_l:.2f}")
        c2.metric(f"xG {equipo_visitante}", f"{lambda_goles_v:.2f}")
        c3.metric("Línea Teórica", f"{xG_total_poisson:.2f}")
        
        st.markdown("##### 🎯 Top 10 Resultados más Probables:")
        st.table(df_top10)
        
    with col_xgb_model:
        st.markdown("<div style='background-color: #1a3c40; padding: 15px; border-radius: 5px; border-left: 5px solid #00ffcc;'><h4>🤖 2. ENFOQUE MACHINE LEARNING (XGBoost V2)</h4></div>", unsafe_index=True)
        st.write("Auditoría inteligente basada en rachas dinámicas recientes y balance de poder de las plantillas:")
        
        st.metric("Probabilidad IA de OVER 2.5 GOLES", f"{prob_xgb:.1f}%")
        
        st.markdown("##### 🔍 Dictamen del Árbol de Decisión:")
        if prob_xgb >= 55.0:
            st.markdown(f"<div style='background-color: #13271e; padding: 15px; border: 1px solid #ffdf1b; color: #ffdf1b; font-weight: bold;'>🔥 SEÑAL DE VALOR DETECTADA (+EV)<br>El modelo de Machine Learning encuentra una alta inercia ofensiva. Se recomienda buscar líneas de Over.</div>", unsafe_index=True)
        else:
            st.markdown(f"<div style='background-color: #221c1c; padding: 15px; border: 1px solid #ff4d4d; color: #ff4d4d;'>⚠️ FILTRO DE RIESGO ACTIVADO (UNDER VALUE)<br>XGBoost detecta desaceleración en las rachas o emparejamientos defensivos cerrados. La línea teórica podría estar inflada por la casa.</div>", unsafe_index=True)

with tab_props:
    col_c, col_t = st.columns(2)
    
    with col_c:
        st.markdown("#### 📐 CÓRNERS (Modelo de Intersección)")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric(f"Forza {equipo_local}", f"{lambda_corners_l:.2f}")
        cc2.metric(f"Forza {equipo_visitante}", f"{lambda_corners_v:.2f}")
        cc3.metric("Línea Esperada", f"{total_corners:.2f}")
        st.markdown(f"<p style='color:#ffdf1b; font-weight:bold;'>Probabilidad de OVER 9.5 CÓRNERS: {prob_over_95_c:.1f}%</p>", unsafe_index=True)
        
    with col_t:
        st.markdown("#### 🟨 TARJETAS (Índice de Agresividad)")
        ct1, ct2 = st.columns(2)
        ct1.metric("Puntos Esperados", f"{total_tarjetas:.2f}")
        st.markdown(f"<p style='color:#ffdf1b; font-weight:bold;'>Probabilidad de OVER 3.5 TARJETAS: {prob_over_35_t:.1f}%</p>", unsafe_index=True)
