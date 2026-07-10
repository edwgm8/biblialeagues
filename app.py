import streamlit as st
import numpy as np
import scipy.stats as stats
import xgboost as xgb
import json
import pandas as pd

# ==============================================================================
# 🎨 ESTILO DE PRODUCCIÓN PREMIUM (BET365 / TRADING WORKSTATION)
# ==============================================================================
st.set_page_config(page_title="Tablero Máster de Probabilidades", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .stApp {
        background-color: #0d1b15;
        color: #ffffff;
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    hr {
        border-top: 1px solid #1f3a30 !important;
    }
    /* Estilos para las tablas de mercados principales */
    .main-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 25px;
    }
    .main-table th {
        background-color: #091410;
        color: #a3b8b0;
        text-align: left;
        padding: 10px;
        font-size: 12px;
        border-bottom: 2px solid #1f3a30;
    }
    .main-table td {
        padding: 12px 10px;
        border-bottom: 1px solid #1f3a30;
        font-size: 14px;
    }
    /* Estilo de matriz estocástica */
    .matrix-table {
        border-collapse: separate;
        border-spacing: 3px;
        width: 100%;
    }
    .matrix-cell {
        text-align: center;
        padding: 15px 5px;
        font-size: 13px;
        font-weight: bold;
        color: #ffffff;
        border-radius: 3px;
    }
    .matrix-header {
        text-align: center;
        color: #a3b8b0;
        font-size: 12px;
        padding: 5px;
    }
    /* Paneles de mercados adicionales */
    .market-box {
        background-color: #13271e;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #ffdf1b;
    }
    .market-title {
        color: #ffdf1b !important;
        font-size: 13px;
        font-weight: bold;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .market-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        font-size: 14px;
    }
    .market-value {
        color: #00ffcc;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 💾 NÚCLEO DE DATOS E INYECCIÓN DE CLUBES NUEVOS
# ==============================================================================
@st.cache_data
def cargar_y_reparar_base_datos():
    with open("bayes_data_6years.json", "r") as f:
        data = json.load(f)
    
    equipos_actuales_2026 = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich", 
        "Leicester", "Liverpool", "Man City", "Man United", "Newcastle", 
        "Nottingham", "Southampton", "Tottenham", "West Ham", "Wolves"
    ]
    
    # Red de seguridad para recién ascendidos
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
    
    for eq in equipos_actuales_2026:
        if eq not in data["teams"]:
            data["teams"][eq] = {
                "attack": round(avg_attack, 2), "defense": round(avg_defense, 2),
                "corners_att": round(avg_corners_att, 2), "corners_def": round(avg_corners_def, 2),
                "cards_recv": round(avg_cards, 2)
            }
    return data

@st.cache_resource
def cargar_modelo_xgb():
    modelo = xgb.XGBClassifier()
    modelo.load_model("xgboost_goles_model.json")
    return modelo

db = cargar_y_reparar_base_datos()
modelo_xgb = cargar_modelo_xgb()
lista_equipos = sorted(list(db["teams"].keys()))

# Controles de Selección
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    eq_l = st.selectbox("🏟️ EQUIPO LOCAL:", lista_equipos, index=lista_equipos.index("Arsenal") if "Arsenal" in lista_equipos else 0)
with col_sel2:
    eq_v = st.selectbox("✈️ EQUIPO VISITANTE:", lista_equipos, index=lista_equipos.index("Man City") if "Man City" in lista_equipos else 1)

if eq_l == eq_v:
    st.warning("Selecciona dos rivales diferentes.")
    st.stop()

# ==============================================================================
# 🧮 CÁLCULO MULTI-MODELO (MATRICES ESTOCÁSTICAS)
# ==============================================================================
data_l, data_v = db["teams"][eq_l], db["teams"][eq_v]
intercept, home_effect = db["intercept"], db["home_effect"]

# 1. MODELO POISSON / BAYES
lambda_l_bayes = np.exp(intercept + home_effect + data_l["attack"] - data_v["defense"])
lambda_v_bayes = np.exp(intercept - home_effect + data_v["attack"] - data_l["defense"])

matrix_bayes = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        matrix_bayes[i, j] = stats.poisson.pmf(i, lambda_l_bayes) * stats.poisson.pmf(j, lambda_v_bayes)

# 2. MODELO XGBOOST (Auditoría de Over 2.5 y calibración de distorsión)
vector = pd.DataFrame([{'home_effect_global': home_effect, 'idx_ataque_local': data_l["attack"], 'idx_defensa_local': data_l["defense"], 'idx_ataque_visita': data_v["attack"], 'idx_defensa_visita': data_v["defense"]}])
prob_over25_xgb = modelo_xgb.predict_proba(vector)[0][1]

# Calibramos la matriz de XGBoost ajustando el xG teórico para que converja con la probabilidad de Over del ML
prob_over25_bayes = 1 - (matrix_bayes[0,0] + matrix_bayes[0,1] + matrix_bayes[0,2] + matrix_bayes[1,0] + matrix_bayes[1,1] + matrix_bayes[2,0])
factor_ajuste = prob_over25_xgb / max(prob_over25_bayes, 0.01)

lambda_l_xgb = lambda_l_bayes * np.sqrt(factor_ajuste)
lambda_v_xgb = lambda_v_bayes * np.sqrt(factor_ajuste)

matrix_xgb = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        matrix_xgb[i, j] = stats.poisson.pmf(i, lambda_l_xgb) * stats.poisson.pmf(j, lambda_v_xgb)

# 3. ENSAMBLE COMBINADO (50% Bayes + 50% XGBoost)
matrix_combinado = (matrix_bayes + matrix_xgb) / 2
lambda_l_comb = (lambda_l_bayes + lambda_l_xgb) / 2
lambda_v_comb = (lambda_v_bayes + lambda_v_xgb) / 2

# Función auxiliar para extraer métricas 1X2 desde cualquier matriz
def procesar_metricas_mercado(matrix, l_local, l_visita):
    p_l = np.sum(np.triu(matrix, 1).T) # i > j
    p_e = np.sum(np.diag(matrix))      # i == j
    p_v = np.sum(np.tril(matrix, -1).T) # i < j
    return f"{p_l*100:.1f}%", f"{p_e*100:.1f}%", f"{p_v*100:.1f}%", f"{l_local:.2f} vs {l_visita:.2f}"

# ==============================================================================
# 📊 RENDERING DE LA TABLA MÁSTER (1X2)
# ==============================================================================
st.markdown("### 📊 Tabla de Probabilidades de Mercados Principales (1X2)")

m_comb = procesar_metricas_mercado(matrix_combinado, lambda_l_comb, lambda_v_comb)
m_xgb = procesar_metricas_mercado(matrix_xgb, lambda_l_xgb, lambda_v_xgb)
m_bayes = procesar_metricas_mercado(matrix_bayes, lambda_l_bayes, lambda_v_bayes)

html_table = f"""
<table class="main-table">
    <tr>
        <th>MÉTODO DE CÁLCULO</th>
        <th>GANA {eq_l.upper()} (1)</th>
        <th>EMPATE (X)</th>
        <th>GANA {eq_v.upper()} (2)</th>
        <th>GOL ESPERADO (xG)</th>
    </tr>
    <tr>
        <td>💛 <b>Ensamble Combinado</b></td>
        <td><b>{m_comb[0]}</b></td><td><b>{m_comb[1]}</b></td><td><b>{m_comb[2]}</b></td><td>{m_comb[3]}</td>
    </tr>
    <tr>
        <td>🌲 XGBoost Híbrido (Rachas)</td>
        <td>{m_xgb[0]}</td><td>{m_xgb[1]}</td><td>{m_xgb[2]}</td><td>{m_xgb[3]}</td>
    </tr>
    <tr>
        <td>🏛️ PyMC Bayes (Jerarquía)</td>
        <td>{m_bayes[0]}</td><td>{m_bayes[1]}</td><td>{m_bayes[2]}</td><td>{m_bayes[3]}</td>
    </tr>
</table>
"""
st.markdown(html_table, unsafe_allow_html=True)

# ==============================================================================
# 🎛️ FILTRO DE ENFOQUE ANALÍTICO ACTIVO
# ==============================================================================
st.write("Filtro de Enfoque Analítico Activo")
enfoque = st.radio("", ["Combinado", "XGBoost", "PyMC Bayes"], horizontal=True, label_visibility="collapsed")

# Asignación de matriz activa según el filtro
if enfoque == "Combinado":
    matriz_activa = matrix_combinado
elif enfoque == "XGBoost":
    matriz_activa = matrix_xgb
else:
    matriz_activa = matrix_bayes

# ==============================================================================
# 🧩 BLOQUE INFERIOR DINÁMICO (MATRIZ VS COLUMNA DE PROPUESTAS)
# ==============================================================================
col_izq, col_der = st.columns([1.3, 1])

with col_izq:
    st.markdown("### 🟥 Matriz Estocástica de Marcadores (%)")
    
    # Generar tabla HTML responsiva para la matriz térmica (Cerrada a 6x6, de 0 a 5 goles)
    grid_html = '<table class="matrix-table">'
    # Renderizado de filas de arriba hacia abajo (Goles Local de 5 a 0)
    for i in reversed(range(6)):
        grid_html += "<tr>"
        grid_html += f'<td style="color:#a3b8b0; font-size:11px; font-weight:bold; width:45px;">{eq_l[:3].upper()} {i}</td>'
        for j in range(6):
            val = matriz_activa[i, j] * 100
            # Paleta dinámica verde-amarilla según probabilidad
            if val > 8.0:
                bg = f"rgba(255, 223, 27, {min(val/12, 1.0)})"
                color = "#000000"
            else:
                bg = f"rgba(19, 150, 91, {min(val/6, 1.0)})"
                color = "#ffffff"
                
            grid_html += f'<td class="matrix-cell" style="background-color: {bg}; color: {color};">{val:.1f}%</td>'
        grid_html += "</tr>"
        
    # Fila de cabecera inferior (Goles Visitante de 0 a 5)
    grid_html += "<tr><td></td>"
    for j in range(6):
        grid_html += f'<td class="matrix-header">{eq_v[:3].upper()} {j}</td>'
    grid_html += "</tr></table>"
    
    st.markdown(grid_html, unsafe_allow_html=True)

with col_der:
    st.markdown("### 📊 Probabilidades de Mercados Adicionales")
    
    # Cálculos derivados del enfoque seleccionado
    p_btts_si = sum(matriz_activa[i, j] for i in range(1, 6) for j in range(1, 6)) * 100
    p_btts_no = 100 - p_btts_si
    
    p_over15 = sum(matriz_activa[i, j] for i in range(6) for j in range(6) if i+j > 1.5) * 100
    p_over25 = sum(matriz_activa[i, j] for i in range(6) for j in range(6) if i+j > 2.5) * 100
    p_over35 = sum(matriz_activa[i, j] for i in range(6) for j in range(6) if i+j > 3.5) * 100
    
    html_sidebar = f"""
    <div class="market-box">
        <div class="market-title">🌍 AMBOS EQUIPOS ANOTAN (BTTS)</div>
        <div class="market-row"><span>Sí (Both Teams to Score - Yes)</span><span class="market-value">{p_btts_si:.1f}%</span></div>
        <div class="market-row"><span>No (Both Teams to Score - No)</span><span class="market-value">{p_btts_no:.1f}%</span></div>
    </div>
    <div class="market-box">
        <div class="market-title">📊 TOTALES DE GOLES (OVER / UNDER)</div>
        <div class="market-row"><span>Más de 1.5 (Over 1.5)</span><span class="market-value">{p_over15:.1f}%</span></div>
        <div class="market-row"><span>Menos de 1.5 (Under 1.5)</span><span class="market-value">{100-p_over15:.1f}%</span></div>
        <div class="market-row" style="background: rgba(255,223,27,0.1); padding: 2px 0;">
            <span style="color:#ffdf1b; font-weight:bold;">Más de 2.5 (Over 2.5)</span><span style="color:#ffdf1b; font-weight:bold;">{p_over25:.1f}%</span>
        </div>
        <div class="market-row"><span>Menos de 2.5 (Under 2.5)</span><span class="market-value">{100-p_over25:.1f}%</span></div>
        <div class="market-row"><span>Más de 3.5 (Over 3.5)</span><span class="market-value">{p_over35:.1f}%</span></div>
        <div class="market-row"><span>Menos de 3.5 (Under 3.5)</span><span class="market-value">{100-p_over35:.1f}%</span></div>
    </div>
    """
    st.markdown(html_sidebar, unsafe_allow_html=True)
    
    # Tabla Inferior: Top 10 Scores Ordenados
    st.markdown("#### 🎯 Top 10 Proyecciones de Score Exacto")
    top_lista = []
    for i in range(6):
        for j in range(6):
            top_lista.append({"Score": f"{eq_l} {i} - {j} {eq_v}", "P": matriz_activa[i, j] * 100})
    df_top = pd.DataFrame(top_lista).sort_values(by="P", ascending=False).head(10).reset_index(drop=True)
    
    df_top.index += 1
    df_top.columns = ["SCORE PROBABLE", "PROBABILIDAD MKT"]
    df_top["PROBABILIDAD MKT"] = df_top["PROBABILIDAD MKT"].map("{:.1f}%".format)
    st.table(df_top)
