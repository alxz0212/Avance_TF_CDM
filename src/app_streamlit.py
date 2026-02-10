import streamlit as st
import streamlit.components.v1 as components
import re
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Configuración de la página
st.set_page_config(
    page_title="Gran Juego Dashboard",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -----------------------------------------------------------------------------
# AUTENTICACIÓN (LOGIN SCREEN)
# -----------------------------------------------------------------------------
def check_password():
    """Returns `True` if the user had the correct password."""

    # Inicializar estado
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] == "Z2456962S" and st.session_state["password"] == "123456A":
            st.session_state["password_correct"] = True
            # Limpiar credenciales de la UI por seguridad
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # Cargar fondo personalizado
    import base64
    def get_base64(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    
    bg_img_path = "/home/jovyan/work/src/static/login_bg_v2.png"
    bg_css = ""
    try:
        bin_str = get_base64(bg_img_path)
        bg_css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stSidebar"] {{ display: none; }}
        [data-testid="stHeader"] {{ visibility: hidden; }}
        
        .login-header {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        }}
        /* Estilar los inputs para que se vean integrados */
        .stTextInput > div > div > input {{
            background-color: transparent; 
            border: none;
            border-bottom: 2px solid #ccc;
            border-radius: 0;
        }}
        </style>
        """
    except:
        pass

    # Renderizar estilos y contenedor visual
    st.markdown(bg_css, unsafe_allow_html=True)
    
    # Columnas para posicionar el formulario en el espacio blanco (derecha)
    # Ajustamos para centrar en la mitad derecha (aprox 60% espacio, 30% form, 10% margen)
    # Usuario pidió "más a la derecha y centrado" -> Aumentamos ratio izq a 2.2
    col_left, col_form, col_right = st.columns([2.2, 0.6, 0.9])

    with col_form:
        # Espacio superior para bajar los inputs y centrarlos verticalmente
        st.markdown("<br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True) 
        st.text_input("Usuario", key="username", value="Z2456962S", placeholder="Nombre de usuario")
        st.text_input("Contraseña", type="password", key="password", value="123456A", placeholder="Contraseña")
        
        if st.button("Acceder", on_click=password_entered):
             if not st.session_state["password_correct"]:
                st.error("Usuario o contraseña incorrectos")

if not check_password():
    st.stop()

# Nota: Streamlit sirve archivos static automáticamente si existen en la carpeta 'static' junto al script.

# Estilos CSS personalizados
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .stApp {
        background-image: linear-gradient(to right top, #ffffff, #f8f9fa, #f1f3f5, #e9ecef, #dee2e6);
    }
    
    /* Optimización para Impresión (PDF) */
    @media print {
        /* Ocultar elementos de UI no deseados */
        [data-testid="stSidebar"], 
        header, 
        footer, 
        .stDeployButton {
            display: none !important;
        }
        /* Ajustar contenido al ancho completo */
        .main .block-container {
            max-width: 100% !important;
            padding: 1rem !important;
        }
        /* Evitar cortes feos en gráficos */
        .stPlotlyChart {
            break-inside: avoid;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. Carga de Datos
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Ruta mapeada en Docker
    DATA_PATH = "/home/jovyan/work/data/processed/qog_great_game.parquet"
    try:
        # Usamos Pandas para agilidad en el dashboard (dataset pequeño)
        df = pd.read_parquet(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

df = load_data()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2. Sidebar y Filtros
st.sidebar.markdown("---")
st.sidebar.header("🎵 Música de Fondo")

# Ruta al archivo de audio
audio_path = "/home/jovyan/work/src/static/spy_glass.mp3"

# Leer el archivo de audio
try:
    import base64
    if os.path.exists(audio_path):
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        # HTML/JS con listener para la barra espaciadora
        audio_html_keybinding = f"""
            <audio id="bg-music" controls>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            <script>
                var audio = document.getElementById("bg-music");
                audio.volume = 0.2; // Volumen inicial

                // Listener para la tecla 'M' (Music)
                document.addEventListener('keydown', function(e) {{
                    // Usar 'm' o 'M' para evitar conflictos con el scroll (espacio)
                    if (e.code === 'KeyM') {{
                        if (audio.paused) {{
                            audio.play();
                        }} else {{
                            audio.pause();
                        }}
                    }}
                }});
            </script>
            <div style="font-size: 0.8em; color: gray; margin-top: 5px;">
                🎧 <i>Spy Glass</i> (Kevin MacLeod)<br>
                <small>💡 Tip: Pulsa <b>M</b> para Play/Pause</small>
            </div>
        """
        st.sidebar.markdown(audio_html_keybinding, unsafe_allow_html=True)
    else:
        st.sidebar.error("No se encontró el archivo de audio.")
except Exception as e:
    st.sidebar.error(f"Error al cargar audio: {e}")

# -----------------------------------------------------------------------------
st.sidebar.title("🌏 Configuración v2.4")
st.sidebar.markdown("---")

if not df.empty:
    # Asegurar años enteros y ordenados
    years = sorted(df['year'].dropna().unique().astype(int))
    
    # Lógica inteligente: Por defecto, seleccionar el último año que tenga datos reales
    # para las variables clave (evita mostrar el 2023 si está vacío).
    valid_data_years = df.dropna(subset=['wdi_expmil', 'gle_cgdpc'])['year'].unique().astype(int)
    if len(valid_data_years) > 0:
        default_year = int(max(valid_data_years))
    else:
        default_year = int(years[-1])

    selected_year = st.sidebar.slider("Filtrar por Año", int(years[0]), int(years[-1]), default_year)
    
    countries = df['cname'].unique()
    selected_countries = st.sidebar.multiselect("Seleccionar Países", countries, default=countries)
    
    # Datos filtrados
    df_filtered = df[(df['year'] == selected_year) & (df['cname'].isin(selected_countries))]
else:
    st.sidebar.warning("No hay datos cargados.")
    df_filtered = pd.DataFrame()

# -----------------------------------------------------------------------------
# 3. Layout Principal
# -----------------------------------------------------------------------------
st.title("🌏 Dashboard: El 'Gran Juego' Post-Soviético")
st.markdown("### Análisis de Factores de Poder y Desarrollo Económico")
st.markdown("---")

# Métricas Clave (KPIs)
col1, col2, col3, col4 = st.columns(4)
if not df_filtered.empty:
    avg_gdp = df_filtered['gle_cgdpc'].mean()
    avg_mil = df_filtered['wdi_expmil'].mean()
    avg_dem = df_filtered['p_polity2'].mean()
    avg_corr = df_filtered['vdem_corr'].mean()
    
    col1.metric("Promedio PIB (PPP)", f"${avg_gdp:,.0f}")
    col2.metric("Gasto Militar (% PIB)", f"{avg_mil:.2f}%")
    col3.metric("Índice Democracia", f"{avg_dem:.1f}")
    col4.metric("Control Corrupción", f"{avg_corr:.2f}")

# Tabs de contenido
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Análisis Exploratorio", "🤖 Modelo ML Interactivo", "🗺️ Visión Regional", "📂 Documentación", "🤖 Asistente IA"])

# -----------------------------------------------------------------------------
# Tab 1: Análisis Exploratorio
# -----------------------------------------------------------------------------
with tab1:
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("Evolución del PIB per Cápita")
        # Line chart de toda la serie histórica (no solo el año filtrado) para los países seleccionados
        df_hist = df[df['cname'].isin(selected_countries)]
        fig_line = px.line(df_hist, x='year', y='gle_cgdpc', color='cname', 
                           markers=True, title="Trayectoria Económica (1991-2023)",
                           labels={'gle_cgdpc': 'PIB per Cápita', 'year': 'Año', 'cname': 'País'})
        
        # Agregamos una línea vertical para indicar el año seleccionado en el slider
        fig_line.add_vline(x=selected_year, line_width=2, line_dash="dash", line_color="red", 
                           annotation_text=f"Año {selected_year}", annotation_position="top right")
                           
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption("ℹ️ **Interpretación:** Visualiza la tendencia histórica del desarrollo económico. La línea vertical roja indica el punto temporal seleccionado para el análisis comparativo.")
        
    with col_viz2:
        st.subheader("Relación: Gasto Militar vs PIB")
        # Limpiar datos para evitar error de NaNs en 'size'
        df_scatter = df_filtered.dropna(subset=['wdi_pop', 'wdi_expmil', 'gle_cgdpc']).copy()
        
        if not df_scatter.empty:
            fig_scatter = px.scatter(df_scatter, x='wdi_expmil', y='gle_cgdpc', 
                                    size='wdi_pop', color='cname', hover_name='cname',
                                    title=f"Scatter Plot (Año {selected_year})",
                                    labels={'wdi_expmil': 'Gasto Militar (%)', 'gle_cgdpc': 'PIB', 'wdi_pop': 'Población'})
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("ℹ️ **Interpretación:** Correlaciona el 'Poder Duro' (inversión militar) con la riqueza nacional. El tamaño de las burbujas representa la Población, añadiendo una dimensión demográfica al análisis.")
        else:
            st.warning("No hay datos completos de Población/Gasto Militar para este año.")

    st.subheader("Matriz de Correlación (Histórico - Países Seleccionados)")
    if not df.empty:
        # Calcular correlación sobre TODO el histórico de los países seleccionados
        # (No filtramos por año porque necesitamos N grande para correlaicón)
        df_corr_source = df[df['cname'].isin(selected_countries)]
        
        features = ['gle_cgdpc', 'wdi_lifexp', 'p_polity2', 'vdem_corr', 'wdi_expmil']
        corr_matrix = df_corr_source[features].dropna().corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                             title=f"Correlación ({', '.join(selected_countries)})")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("""
        ℹ️ **Interpretación de la Matriz:**
        1. **Democracia vs Corrupción:** Existe una notable **correlación negativa** (aprox. -0.6). Esto sugiere que los países con mayores índices democráticos (`p_polity2`) tienden a tener menores niveles de corrupción (`vdem_corr`).
        2. **Economía y Bienestar:** El PIB per cápita (`gle_cgdpc`) tiene una **correlación positiva** con la Esperanza de Vida (`wdi_lifexp`), confirmando que el desarrollo económico impulsa la longevidad.
        3. **Poder Militar:** El Gasto Militar (`wdi_expmil`) correlaciona positivamente con el PIB, lo que indica que las economías más fuertes de la región tienen mayor capacidad para financiar sus fuerzas armadas (Poder Duro).
        """)

# -----------------------------------------------------------------------------
# Tab 2: Modelo ML Interactivo
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("""
    ### 🔮 Simulador Random Forest
    Entrena un modelo en tiempo real y **mueve los deslizadores** para predecir cómo cambiaría el PIB bajo diferentes condiciones políticas.
    """)
    
    col_ml_left, col_ml_right = st.columns([1, 2])
    
    # Entrenar modelo (Scikit-Learn)
    features_ml = ['wdi_lifexp', 'p_polity2', 'vdem_corr', 'wdi_expmil']
    target_ml = 'gle_cgdpc'
    
    df_ml = df.dropna(subset=features_ml + [target_ml])
    X = df_ml[features_ml]
    y = df_ml[target_ml]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    
    with col_ml_left:
        st.success(f"Modelo Entrenado (R²: {r2:.2f})")
        st.markdown("#### Parámetros de Simulación:")
        
        sim_life = st.slider("Esperanza de Vida", float(X['wdi_lifexp'].min()), float(X['wdi_lifexp'].max()), float(X['wdi_lifexp'].mean()))
        sim_dem = st.slider("Democracia (Polity)", -10.0, 10.0, float(X['p_polity2'].mean()))
        sim_corr = st.slider("Control Corrupción", 0.0, 1.0, float(X['vdem_corr'].mean()))
        sim_mil = st.slider("Gasto Militar (%)", 0.0, float(X['wdi_expmil'].max()), float(X['wdi_expmil'].mean()))
        
        input_data = pd.DataFrame([[sim_life, sim_dem, sim_corr, sim_mil]], columns=features_ml)
        prediction = model.predict(input_data)[0]
        
        st.metric("PIB Predicho", f"${prediction:,.2f}")

    with col_ml_right:
        st.subheader("Importancia de Variables (Feature Importance)")
        importances = pd.DataFrame({
            'Feature': features_ml,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', 
                         color='Importance', color_continuous_scale='Viridis',
                         title="Importancia de Variables (Feature Importance)",
                         text_auto='.2f') # Muestra el valor en las barras con 2 decimales
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("ℹ️ **Interpretación ML:** El modelo Random Forest identifica qué variables influyen más en la predicción del PIB. Nótese cómo el Gasto Militar (`wdi_expmil`) a menudo supera a las variables democráticas, validando la hipótesis del 'Poder Duro'.")

# -----------------------------------------------------------------------------
# Tab 3: Visión regional
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Comparativa por Subregiones")
    fig_box = px.box(df, x="subregion", y="gle_cgdpc", color="subregion", 
                     title="Distribución del PIB por Región Geopolítica",
                     points="all")
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("ℹ️ **Interpretación Regional:** Este gráfico de caja (Boxplot) compara la dispersión de la riqueza económica. Permite identificar qué subregión tiene mayor PIB mediano y qué tan desigual es el crecimiento entre los países de cada zona.")

def read_markdown_file(filename):
    path = f"/home/jovyan/work/docs/{filename}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error al leer el archivo {filename}: {e}"

def render_markdown_with_mermaid(markdown_text):
    """
    Renderiza markdown normal y bloques mermaid usando JS.
    """
    if "```mermaid" not in markdown_text:
        st.markdown(markdown_text, unsafe_allow_html=True)
        return

    # Patrón para encontrar bloques mermaid
    # Usamos re.split para separar el texto en [texto, mermaid_code, texto, mermaid_code...]
    parts = re.split(r'```mermaid\n(.*?)\n```', markdown_text, flags=re.DOTALL)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Es markdown normal
            if part.strip():
                st.markdown(part, unsafe_allow_html=True)
        else:
            # Es código mermaid
            mermaid_code = part.strip()
            if mermaid_code:
                # Renderizar HTML con Mermaid.js
                # Usamos un ID único para evitar conflictos si hay múltiples
                
                html_code = f"""
                <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                    mermaid.initialize({{ startOnLoad: true }});
                </script>
                <div class="mermaid" style="background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; border: 1px solid #ddd;">
                    {mermaid_code}
                </div>
                """
                components.html(html_code, height=500, scrolling=True)

with tab4:
    st.header("📂 Documentación del Proyecto")
    st.markdown("Selecciona el documento que deseas visualizar:")
    
    docs = {
        "ℹ️ README (General)": "01_README.md",
        "🏗️ Infraestructura": "02_INFRAESTRUCTURA.md",
        "📊 Resultados y Análisis": "03_RESULTADOS.md",
        "🧠 Reflexión IA": "04_REFLEXION_IA.md",
        "💻 Explicación Código": "05_EXPLICACION_CODIGO.md",
        "📝 Respuestas": "06_RESPUESTAS.md",
        "🎥 Prototipo / Demo": "07_PROTOTIPO.md"
    }
    
    selected_doc_name = st.radio("Archivos Disponibles:", list(docs.keys()), horizontal=True)
    
    st.markdown("---")
    
    # Mostrar contenido del archivo seleccionado
    file_content = read_markdown_file(docs[selected_doc_name])
    
    # Inyectar video si es el PROTOTIPO (Para que se vea en el Dashboard)
    if docs[selected_doc_name] == "07_PROTOTIPO.md":
        # Usamos st.video nativo para evitar problemas de rutas HTML
        # Streamlit resuelve mejor las rutas locales si le pasamos el path del archivo
        # Usamos columnas para centrar y reducir el tamaño del video (Efecto "Zoom" al poner pantalla completa)
        col_spacer1, col_vid, col_spacer2 = st.columns([1, 2, 1])
        with col_vid:
            import os
            video_path = os.path.join(os.path.dirname(__file__), "static", "dashboard_demo.mp4")
            st.video(video_path)
        
        # Eliminamos la inyección manual anterior para no duplicar
        video_html = ""
        # Reemplazar la imagen del GIF por el video real interactivo
        # Buscamos el patrón markdown del GIF: ![...](capturas/dashboard_demo.gif)
        # Si no lo encuentra, lo inserta al principio como fallback
        if "dashboard_demo.gif" in file_content:
            import re
            # Reemplaza cualquier imagen que apunte al gif
            file_content = re.sub(r'!\[.*?\]\(.*?dashboard_demo.gif\)', video_html, file_content)
        else:
            file_content = video_html + file_content

    render_markdown_with_mermaid(file_content)

# -----------------------------------------------------------------------------
# Tab 5: Asistente IA (Algorithmic Analyst)
# -----------------------------------------------------------------------------
with tab5:
    st.header("🤖 Asistente Virtual: 'QoG-Bot'")
    st.markdown("""
    Este asistente utiliza lógica analítica avanzada para generar reportes automáticos y responder preguntas sobre los datos.
    """)
    
    col_bot1, col_bot2 = st.columns([1, 2])
    
    with col_bot1:
        st.subheader("📝 Generar Reporte Automático")
        report_country = st.selectbox("Elige un país para analizar:", df['cname'].unique())
        if st.button("Generar Informe"):
            # Lógica de "AI" narrativa
            country_data = df[df['cname'] == report_country].sort_values('year')
            
            # Helper para buscar dato válido más reciente
            def get_val(data, col):
                valid = data.dropna(subset=[col])
                if not valid.empty:
                    row = valid.iloc[-1]
                    return row[col], int(row['year'])
                return None, None

            gdp, gdp_yr = get_val(country_data, 'gle_cgdpc')
            mil, mil_yr = get_val(country_data, 'wdi_expmil')
            pol, pol_yr = get_val(country_data, 'p_polity2')
            
            # Cálculos comparativos (usando el año del dato encontrado)
            if gdp:
                avg_gdp_region = df[df['year'] == gdp_yr]['gle_cgdpc'].mean()
                status_eco = "superior" if gdp > avg_gdp_region else "inferior"
                gdp_val_fmt = f"{gdp:,.0f} USD"
                gdp_txt = f"{gdp_val_fmt} (dato {gdp_yr})"
                comp_txt = f"{avg_gdp_region:,.0f} USD"
            else:
                status_eco, gdp_txt, comp_txt = "desconocido", "No disponible", "N/A"
                gdp_val_fmt = "N/A"

            trend_dem = "estable"
            if pol is not None:
                first_pol = country_data.iloc[0]['p_polity2']
                if pd.notna(first_pol):
                    trend_dem = "mejorando" if pol > first_pol else "empeorando" if pol < first_pol else "igual"
            
            pol_val = f"{pol}" if pol is not None else "No disponible"
            mil_val = f"{mil:.2f}%" if mil is not None else "No disponible"
            
            wdi_expmil_val = mil if mil is not None else 0.0

            narrative = f"""
            ### 🕵️ Análisis de Inteligencia para **{report_country}**
            
            **1. Situación Económica:**
            El PIB per cápita más reciente es de **{gdp_val_fmt}** (año {gdp_yr}), lo cual es **{status_eco}** al promedio de la región ({comp_txt}).
            
            **2. Perfil de Poder:**
            {report_country} muestra un Gasto Militar del **{mil_val}** del PIB. 
            En términos políticos, su índice democrático es **{pol_val}** (escala -10 a 10), mostrando una tendencia **{trend_dem}**.
            
            **3. Conclusión Algorítmica:**
            Este perfil sugiere un estado que prioriza {'la seguridad (Poder Duro)' if wdi_expmil_val > 3.0 else 'el desarrollo civil/mixto'}.
            """
            st.success("Informe generado con éxito.")
            st.markdown(narrative)
            
    with col_bot2:
        st.subheader("💬 Chat con tus Datos")
        
        # Inicializar historial de chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Mensaje de bienvenida
            st.session_state.messages.append({"role": "assistant", "content": "¡Hola! Soy tu asistente de Big Data. Pregúntame cosas como: '¿Cuál es el país más rico?', '¿Promedio de esperanza de vida?' o 'Dime sobre Afganistán'."})

        # Mostrar mensajes previos
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input de chat
        if prompt := st.chat_input("Escribe tu pregunta aquí..."):
            # Guardar y mostrar mensaje usuario
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Lógica del Bot (Keyword Matching Local)
            prompt_lower = prompt.lower()
            response = "No estoy seguro de entender eso. Prueba preguntando por 'PIB', 'militar', 'democracia' o un país específico."
            
            # 1. Preguntas sobre Máximos
            if "rico" in prompt_lower or "pib" in prompt_lower and "mayor" in prompt_lower:
                # Recuperamos el último año con datos válidos
                valid_df = df.dropna(subset=['gle_cgdpc'])
                if not valid_df.empty:
                    last_valid_year = valid_df['year'].max()
                    df_last_valid = valid_df[valid_df['year'] == last_valid_year]
                    max_country = df_last_valid.loc[df_last_valid['gle_cgdpc'].idxmax()]
                    response = f"El país más rico (mayor PIB per cápita, {int(last_valid_year)}) es **{max_country['cname']}** con ${max_country['gle_cgdpc']:,.2f}."
                else:
                    response = "No hay datos suficientes de PIB."
            
            elif "militar" in prompt_lower and ("mayor" in prompt_lower or "más" in prompt_lower):
                 valid_df = df.dropna(subset=['wdi_expmil'])
                 if not valid_df.empty:
                     last_valid_year = valid_df['year'].max()
                     df_last_valid = valid_df[valid_df['year'] == last_valid_year]
                     max_mil = df_last_valid.loc[df_last_valid['wdi_expmil'].idxmax()]
                     response = f"El país que más gasta en ejército ({int(last_valid_year)}) es **{max_mil['cname']}** con un **{max_mil['wdi_expmil']:.2f}%** de su PIB."
                 else:
                     response = "No hay datos suficientes de Gasto Militar."

            # 2. Preguntas sobre Promedios
            elif "promedio" in prompt_lower:
                if "vida" in prompt_lower:
                    avg_life = df['wdi_lifexp'].mean()
                    response = f"La esperanza de vida promedio en la región (histórico) es de **{avg_life:.1f} años**."
                elif "pib" in prompt_lower:
                    avg_gdp = df['gle_cgdpc'].mean()
                    response = f"El PIB per cápita promedio histórico es de **${avg_gdp:,.2f}**."

            # 3. Preguntas sobre Países Específicos
            elif any(country.lower() in prompt_lower for country in df['cname'].unique().tolist()):
                for country in df['cname'].unique():
                    if country.lower() in prompt_lower:
                        # Obtener datos más recientes de ese país
                        country_df = df[df['cname'] == country].sort_values(by='year', ascending=False)
                        if not country_df.empty:
                            row = country_df.iloc[0]
                            response = (f"**Datos más recientes de {country} ({int(row['year'])}):**\n"
                                        f"- 💰 PIB: ${row['gle_cgdpc']:,.0f}\n"
                                        f"- 🛡️ Gasto Militar: {row['wdi_expmil']:.2f}%\n"
                                        f"- 🩺 Esperanza Vida: {row['wdi_lifexp']:.1f} años")
                        else:
                            response = f"No tengo datos para {country}."
                        break
            
            # 4. Easter Eggs
            elif "hola" in prompt_lower:
                response = "¡Hola! Listo para analizar el Gran Juego."
            elif "gracias" in prompt_lower:
                response = "¡De nada! ¿Alguna otra consulta?"

            # Simular comportamiento "AI Realista"
            import time
            import random
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # 1. Efecto "Pensando..." (Delay inicial)
                message_placeholder.markdown("_(Analizando datos...)_ 🧠")
                time.sleep(random.uniform(1.2, 2.5)) 
                
                # 2. Efecto "Escribiendo" más natural
                full_response = ""
                for chunk in response.split():
                    full_response += chunk + " "
                    # Velocidad variable para parecer más humano/bot generativo
                    time.sleep(random.uniform(0.05, 0.2)) 
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
