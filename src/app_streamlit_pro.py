
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -----------------------------------------------------------------------------
# 1. Configuración de Página & CSS Hacking
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Geopolitics Command Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Diccionario de Banderas y Coordenadas para el Globo 3D
COUNTRY_CONFIG = {
    "Afghanistan": {"flag": "🇦🇫", "lat": 33.9391, "lon": 67.7100, "iso": "AFG"},
    "Mongolia":    {"flag": "🇲🇳", "lat": 46.8625, "lon": 103.8467, "iso": "MNG"},
    "Azerbaijan":  {"flag": "🇦🇿", "lat": 40.1431, "lon": 47.5769, "iso": "AZE"},
    "Georgia":     {"flag": "🇬🇪", "lat": 42.3154, "lon": 43.3569, "iso": "GEO"},
    "Armenia":     {"flag": "🇦🇲", "lat": 40.0691, "lon": 45.0382, "iso": "ARM"}
}

# Inyección de CSS para "Look & Feel" Profesional
st.markdown("""
<style>
    /* Importar fuente futurista */
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Roboto:wght@300;700&display=swap');

    /* Fondo general oscuro */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Headers con fuente mono */
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Métricas estilo HUD */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 31, 38, 0.8);
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #00d4ff; /* Cyan Neon */
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.1);
    }
    
    /* Textos de métricas */
    label[data-testid="stMetricLabel"] {
        color: #888 !important;
        font-family: 'Share Tech Mono', monospace;
    }
    div[data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-family: 'Share Tech Mono', monospace;
        font-size: 28px !important;
    }

    /* Tabs Neon Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1c1f26;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #888;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117 !important;
        color: #00d4ff !important;
        border-top: 2px solid #00d4ff;
    }

    /* Ocultar elementos default de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Optimización para Impresión (PDF) */
    /* Optimización para Impresión (PDF) */
    @media print {
        /* Forzar impresión de fondos (Oscuros) */
        * {
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }

        /* Ocultar elementos de UI */
        [data-testid="stSidebar"], 
        header, 
        footer, 
        .stDeployButton {
            display: none !important;
        }
        
        /* Mantener estilo oscuro en impresión si el usuario lo desea */
        .stApp, .stApp > header {
            background-color: #0e1117 !important;
            color: #e0e0e0 !important;
        }
        
        /* Ajustar ancho para PDF A4/Letter */
        .main .block-container {
            max-width: 100% !important;
            padding: 1rem !important;
        }
        
        /* Evitar cortes en gráficos */
        .stPlotlyChart, div[data-testid="stMetricValue"] {
            break-inside: avoid;
            page-break-inside: avoid;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Carga de Datos
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    DATA_PATH = "/home/jovyan/work/data/processed/qog_great_game.parquet"
    try:
        df = pd.read_parquet(DATA_PATH)
        return df
    except Exception:
        return pd.DataFrame()

df = load_data()

# -----------------------------------------------------------------------------
# 3. Header & HUD (Head-Up Display)
# -----------------------------------------------------------------------------
st.title("📡 GEO-POLITICS COMMAND CENTER")
st.markdown("_Análisis de Inteligencia Estratégica: El Gran Juego Post-Soviético_")
st.markdown("---")

# Filtros Globales (Barra Superior disimulada)
if not df.empty:
    years = sorted(df['year'].unique().astype(int))
    selected_year = st.slider("TIMELINE SELECTOR", min_value=years[0], max_value=years[-1], value=years[-1], label_visibility="collapsed")
    df_yr = df[df['year'] == selected_year]
else:
    df_yr = pd.DataFrame()

# Métricas Globales (HUD)
col1, col2, col3, col4, col5 = st.columns(5)
if not df_yr.empty:
    col1.metric("AÑO ACTIVO", f"{selected_year}", delta=None)
    col2.metric("PIB REGIONAL AVG", f"${df_yr['gle_cgdpc'].mean():,.0f}", delta_color="normal")
    col3.metric("GASTO MILITAR AVG", f"{df_yr['wdi_expmil'].mean():.2f}%", delta_color="normal")
    col4.metric("NIVEL DEMOCRACIA", f"{df_yr['p_polity2'].mean():.1f}", help="Escala de -10 a 10")
    col5.metric("PAÍSES TRACKEADOS", f"{len(df_yr['cname'].unique())}")


# -----------------------------------------------------------------------------
# 4. Funciones de Renderizado (Modularización para Reportes)
# -----------------------------------------------------------------------------
def render_geo_dashboard(df_target):
    st.markdown("## 🌍 GLOBAL SITUATION ROOM")
    row_geo = st.columns([2, 1])
    
    with row_geo[0]:
        st.subheader(f"🗺️ PROYECCIÓN GEOPOLÍTICA ({selected_year})")
        if not df_target.empty:
            # Preparar datos geoespaciales
            map_data = df_target.copy()
            # Añadir lat/lon desde config
            map_data['lat'] = map_data['cname'].map(lambda x: COUNTRY_CONFIG.get(x, {}).get('lat', 0))
            map_data['lon'] = map_data['cname'].map(lambda x: COUNTRY_CONFIG.get(x, {}).get('lon', 0))
            map_data['flag'] = map_data['cname'].map(lambda x: COUNTRY_CONFIG.get(x, {}).get('flag', ''))
            
            # Crear Globo 3D
            fig_globe = px.scatter_geo(map_data, lat='lat', lon='lon',
                                     color="wdi_expmil", size="gle_cgdpc",
                                     hover_name="cname",
                                     projection="orthographic", # Globo terráqueo
                                     color_continuous_scale="Viridis",
                                     title="",
                                     height=600)
            
            # Customizar diseño "Dark"
            fig_globe.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                paper_bgcolor="#0e1117",
                geo=dict(
                    bgcolor="#0e1117",
                    showland=True, landcolor="#1c1f26",
                    showocean=True, oceancolor="#0e1117",
                    showlakes=False,
                    showcountries=True, countrycolor="#444",
                    projection_rotation=dict(lon=60, lat=30), # Centrar en Asia Central
                ),
                font=dict(color="white")
            )
            st.plotly_chart(fig_globe, use_container_width=True)
            
    with row_geo[1]:
        st.subheader("📊 RANKING DE PODER")
        if not df_target.empty:
            top_mil = df_target[['cname', 'wdi_expmil', 'gle_cgdpc']].sort_values('wdi_expmil', ascending=False)
            
            for index, row in top_mil.iterrows():
                flag = COUNTRY_CONFIG.get(row['cname'], {}).get('flag', '')
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #1c1f26; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 3px solid #ff4b4b;">
                        <h4 style="margin:0; color: white;">{flag} {row['cname']}</h4>
                        <small style="color: #aaa;">Gasto Militar: <span style="color: #ff4b4b;">{row['wdi_expmil']:.2f}%</span> | PIB: <span style="color: #00d4ff;">${row['gle_cgdpc']:,.0f}</span></small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.info("ℹ️ El tamaño de las esferas en el globo representa el PIB, el color indica la militarización.")

def render_pvp_analysis(df_target):
    st.markdown("## ⚔️ HEAD-TO-HEAD")
    st.subheader("⚔️ ANÁLISIS COMPARATIVO DIRECTO")
    
    col_sel1, col_sel2 = st.columns(2)
    p1 = col_sel1.selectbox("PAÍS A (Blue Team)", list(COUNTRY_CONFIG.keys()), index=0)
    p2 = col_sel2.selectbox("PAÍS B (Red Team)", list(COUNTRY_CONFIG.keys()), index=1)
    
    if not df_target.empty and p1 and p2:
        # Extraer datos de los paises seleccionados
        d1 = df_target[df_target['cname'] == p1]
        d2 = df_target[df_target['cname'] == p2]
        
        if not d1.empty and not d2.empty:
            # Normalizar para radar chart (Escala 0-1 aproximada para visualización)
            # Esto es solo visual, no estadístico riguroso
            categories = ['Gasto Militar', 'Democracia (Norm)', 'Control Corrupción', 'Esperanza Vida (Norm)', 'PIB (Log)']
            
            def get_values(row):
                # Conversiones "al vuelo" para que quepan en el gráfico de radar
                mil = row['wdi_expmil'].values[0]
                dem = (row['p_polity2'].values[0] + 10) / 2 # Escala 0-10
                corr = row['vdem_corr'].values[0] * 10
                life = (row['wdi_lifexp'].values[0] - 50) / 4 # Ajuste visual
                import numpy as np
                gdp = np.log(row['gle_cgdpc'].values[0])
                return [mil, dem, corr, life, gdp]

            fig_radar = go.Figure()

            fig_radar.add_trace(go.Scatterpolar(
                  r=get_values(d1),
                  theta=categories,
                  fill='toself',
                  name=f"{COUNTRY_CONFIG[p1]['flag']} {p1}",
                  line_color='#00d4ff'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                  r=get_values(d2),
                  theta=categories,
                  fill='toself',
                  name=f"{COUNTRY_CONFIG[p2]['flag']} {p2}",
                  line_color='#ff4b4b'
            ))

            fig_radar.update_layout(
              polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], color="#555"),
                bgcolor="#1c1f26"
              ),
              showlegend=True,
              paper_bgcolor="#0e1117",
              font=dict(color="white"),
              height=500
            )

            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Tabla comparativa rápida
            st.markdown("#### 📋 DATOS CRUDOS")
            comp_df = pd.concat([d1, d2])[['cname', 'gle_cgdpc', 'wdi_expmil', 'p_polity2', 'vdem_corr']]
            st.dataframe(comp_df.style.format({"gle_cgdpc": "${:,.0f}", "wdi_expmil": "{:.2f}%"}))
        else:
            st.warning("Datos no disponibles para uno de los países en este año.")

def render_ai_simulator(df_total):
    st.markdown("## 🤖 AI SIMULATOR")
    col_sim_controls, col_sim_res = st.columns([1, 2])
    
    with col_sim_controls:
        st.markdown("### 🎛️ PANEL DE CONTROL")
        st.markdown("Ajusta los parámetros para simular un escenario hipotético:")
        
        sim_mil = st.slider("🔧 Inversión Militar (%)", 0.0, 10.0, 2.5)
        sim_dem = st.slider("🗳️ Índice Democrático", -10, 10, 5)
        sim_corr = st.slider("⚖️ Control Corrupción", 0.0, 1.0, 0.5)
        sim_life = st.slider("🏥 Esperanza Vida", 50, 85, 70)
        
    with col_sim_res:
        st.markdown("### 🔮 PREDICCIÓN (RANDOM FOREST)")
        
        # Entrenar modelo rápido - CACHED
        @st.cache_resource
        def train_model_pro(data):
            ml_cols = ['wdi_lifexp', 'p_polity2', 'vdem_corr', 'wdi_expmil']
            df_ml = data.dropna(subset=ml_cols + ['gle_cgdpc'])
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(df_ml[ml_cols], df_ml['gle_cgdpc'])
            return rf

        if not df_total.empty:
            rf = train_model_pro(df_total)
            
            # Predecir
            pred = rf.predict([[sim_life, sim_dem, sim_corr, sim_mil]])[0]
            
            # Visualizar resultado tipo "Gauge"
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = pred,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "PIB Per Cápita Proyectado"},
                delta = {'reference': df_total['gle_cgdpc'].mean(), 'increasing': {'color': "#00d4ff"}},
                gauge = {
                    'axis': {'range': [None, 10000], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#00d4ff"},
                    'bgcolor': "#1c1f26",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 3000], 'color': '#ff4b4b'},
                        {'range': [3000, 7000], 'color': '#ffe600'}],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': pred}}))
            
            fig_gauge.update_layout(paper_bgcolor="#0e1117", font={'color': "white", 'family': "Share Tech Mono"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.info(f"ℹ️ Con estos parámetros, el modelo predice una economía de **${pred:,.0f}** por habitante.")

def render_dataset(df_total):
    st.markdown("## 📄 DATA SOURCE")
    with st.expander("Ver Dataset Completo", expanded=True):
        st.dataframe(df_total.style.background_gradient(cmap="viridis"), height=600)
    st.caption("Fuente: Quality of Government (QoG) Standard Time-Series Dataset")


# -----------------------------------------------------------------------------
# 5. Panel Principal (Control de Flujo)
# -----------------------------------------------------------------------------
st.write("")

# Checkbox para Modos Especiales
col_mode_1, col_mode_2 = st.sidebar.columns(2)
print_mode = col_mode_1.checkbox("🖨️ Imprimir", value=False, help="Modo Reporte PDF")
pres_mode = col_mode_2.checkbox("📺 Presentación", value=False, help="Modo Diapositivas Interactivas")

if pres_mode:
    # --- MODO PRESENTACIÓN INTERACTIVO ---
    if "slide_index" not in st.session_state:
        st.session_state.slide_index = 0

    # Definir las 'Diapositivas' (Funciones + Títulos)
    slides = [
        {"title": "🌍 1. GLOBAL SITUATION ROOM", "func": lambda: render_geo_dashboard(df_yr)},
        {"title": "⚔️ 2. HEAD-TO-HEAD ANALYSIS", "func": lambda: render_pvp_analysis(df_yr)},
        {"title": "🤖 3. AI STRATEGIC SIMULATOR", "func": lambda: render_ai_simulator(df)},
        {"title": "📄 4. DATA SOURCE INTELLIGENCE", "func": lambda: render_dataset(df)}
    ]

    total_slides = len(slides)
    current_slide = st.session_state.slide_index

    # Navegación
    st.markdown("---")
    col_prev, col_info, col_next = st.columns([1, 4, 1])
    
    with col_prev:
        if st.button("⬅️ PREV", use_container_width=True):
            st.session_state.slide_index = max(0, current_slide - 1)
            st.rerun()
    
    with col_info:
        st.markdown(f"<h3 style='text-align: center; margin: 0;'>{slides[current_slide]['title']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: gray;'>SLIDE {current_slide + 1} / {total_slides}</p>", unsafe_allow_html=True)

    with col_next:
        if st.button("NEXT ➡️", use_container_width=True):
            st.session_state.slide_index = min(total_slides - 1, current_slide + 1)
            st.rerun()

    st.markdown("---")
    
    # Renderizar contenido de la slide actual
    slides[current_slide]["func"]()

elif print_mode:
    # MODO IMPRESIÓN: Renderizar todo secuencialmente
    st.info("🖨️ **MODO IMPRESIÓN ACTIVADO:** Todo el contenido se muestra en una sola página larga. Usa Ctrl+P (o Command+P) para guardar como PDF. Asegúrate de activar 'Gráficos de fondo' en las opciones de impresión.")
    
    st.markdown("---")
    render_geo_dashboard(df_yr)
    st.markdown("---")
    render_pvp_analysis(df_yr)
    st.markdown("---")
    render_ai_simulator(df)
    st.markdown("---")
    render_dataset(df)

else:
    # MODO STANDARD: Renderizar con Tabs
    tab_geo, tab_vs, tab_ai, tab_raw = st.tabs(["🌍 GLOBAL SITUATION ROOM", "⚔️ HEAD-TO-HEAD", "🤖 AI SIMULATOR", "📄 DATA SOURCE"])

    # --- TAB 1: SITUATION ROOM ---
    with tab_geo:
        render_geo_dashboard(df_yr)

    # --- TAB 2: HEAD-TO-HEAD ---
    with tab_vs:
        render_pvp_analysis(df_yr)

    # --- TAB 3: AI SIMULATOR ---
    with tab_ai:
        render_ai_simulator(df)

    # --- TAB 4: DATA SOURCE ---
    with tab_raw:
        render_dataset(df)
