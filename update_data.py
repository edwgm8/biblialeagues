import requests
import json
import numpy as np

# ==============================================================================
# 🔑 CONFIGURACIÓN DE CREDENCIALES Y SETTINGS
# ==============================================================================
API_TOKEN = "be9935c7689f4f48aaf1e44ea51fe65b"
API_URL = "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED"
JSON_FILE = "bayes_data_6years.json"

# Mapeo obligatorio: La API entrega los nombres con "FC" o variaciones.
# Esto los traduce exactamente a como los tienes en tu ecosistema.
NAME_MAPPER = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry City",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City FC": "Hull City",
    "Ipswich Town FC": "Ipswich Town",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nottingham",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves"
}

# FACTOR DE APRENDIZAJE (Indica qué tan rápido cambian los índices con los nuevos partidos)
LEARNING_RATE = 0.05 

def ejecutar_actualizacion():
    print("🔄 Iniciando proceso de actualización desde Football-Data.org...")
    
    # 1. Cargar tu base de datos actual
    try:
        with open(JSON_FILE, "r") as f:
            db = json.load(f)
    except Exception as e:
        print(f"❌ Error al leer {JSON_FILE}: {e}")
        return

    # 2. Realizar petición segura a la API
    headers = {"X-Auth-Token": API_TOKEN}
    response = requests.get(API_URL, headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Error de conexión con la API. Código de estado: {response.status_code}")
        return
        
    data = response.json()
    matches = data.get("matches", [])
    print(f"⚽ Se encontraron {len(matches)} partidos finalizados en la API.")

    if not matches:
        print("ℹ️ No hay partidos nuevos por procesar.")
        return

    partidos_procesados = 0

    # 3. Procesar cada partido finalizado para ajustar índices de goles
    for match in matches:
        api_home = match["homeTeam"]["name"]
        api_away = match["awayTeam"]["name"]
        
        # Filtrar si los equipos están en nuestro mapa de la Premier
        if api_home in NAME_MAPPER and api_away in NAME_MAPPER:
            team_l = NAME_MAPPER[api_home]
            team_v = NAME_MAPPER[api_away]
            
            # Obtener goles reales del partido
            goles_l = match["score"]["fullTime"]["home"]
            goles_v = match["score"]["fullTime"]["away"]
            
            if goles_l is None or goles_v is None:
                continue
                
            # Traer los índices actuales de tu JSON
            idx_att_l = db["teams"][team_l]["attack"]
            idx_def_l = db["teams"][team_l]["defense"]
            idx_att_v = db["teams"][team_v]["attack"]
            idx_def_v = db["teams"][team_v]["defense"]
            
            # ALGORITMO DE AJUSTE DINÁMICO (Alineación de Fuerza Ofensiva / Defensiva)
            # Si anotan más goles de lo esperado contractualmente frente al rival, sus índices mejoran.
            xg_teorico_l = np.exp(db["intercept"] + db["home_effect"] + idx_att_l - idx_def_v)
            xg_teorico_v = np.exp(db["intercept"] - db["home_effect"] + idx_att_v - idx_def_l)
            
            # Ajuste de ataque del local y defensa del visitante
            error_l = goles_l - xg_teorico_l
            db["teams"][team_l]["attack"] = round(idx_att_l + (error_l * LEARNING_RATE), 3)
            db["teams"][team_v]["defense"] = round(idx_def_v - (error_l * LEARNING_RATE), 3)
            
            # Ajuste de ataque del visitante y defensa del local
            error_v = goles_v - xg_teorico_v
            db["teams"][team_v]["attack"] = round(idx_att_v + (error_v * LEARNING_RATE), 3)
            db["teams"][team_l]["defense"] = round(idx_def_l - (error_v * LEARNING_RATE), 3)
            
            partidos_procesados += 1

    # 4. Guardar los nuevos índices calibrados en tu JSON máster
    with open(JSON_FILE, "w") as f:
        json.dump(db, f, indent=4)
        
    print(f"✅ ¡Actualización completada con éxito! Se calibraron {partidos_procesados} partidos en vivo.")

if __name__ == "__main__":
    ejecutar_actualizacion()
