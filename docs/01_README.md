# Trabajo Final: El "Gran Juego" Post-Soviético

**Alumno:** Daniel Alexis Mendoza Corne  
**Fecha:** Febrero 2026

---

---

## 🎥 Demostración

**¿Quieres ver cómo quedó el Dashboard?** 👉 [Mira el video aqui](07_PROTOTIPO.md)

---

## Orden de trabajo

Completa los archivos en este orden. Cada numero indica la secuencia:

| Orden  | Archivo                       | Que haces                                   |
| ------ | ----------------------------- | ------------------------------------------- |
| **1**  | `01_README.md` (este archivo) | Defines tu pregunta, paises y variables     |
| **2**  | `02_INFRAESTRUCTURA.md`       | Construyes y explicas tu docker-compose.yml |
| **3**  | `src/verify_spark.py`         | Verificas la conexión con Spark             |
| **4**  | `src/pipeline.py`             | ETL: Limpieza y Transformación en Parquet   |
| **5**  | `src/analysis.py`             | Análisis con ML (Random Forest) en Spark    |
| **6**  | `src/econometric_analysis.py` | Análisis Econométrico (Test de Hausman)     |
| **7**  | `03_RESULTADOS.md`            | Presentas graficos e interpretas resultados |
| **8**  | `04_REFLEXION_IA.md`          | Documentas tu proceso y pegas tus prompts   |
| **9**  | `05_EXPLICACION_CODIGO.md`    | Catálogo técnico de todos los scripts       |
| **10** | `06_RESPUESTAS.md`            | Respondes 4 preguntas de comprension        |
| **11** | `07_PROTOTIPO.md`             | **Nuevo:** Video Demo del Dashboard         |

Los archivos `docker-compose.yml`, `requirements.txt` y `.gitignore` los completas conforme avanzas.

---

## Pregunta de investigacion

"¿Qué influye más en la riqueza de los países ex-soviéticos: tener un ejército fuerte y gastar mucho en armas, o ser un país más democrático y con menos corrupción?"

---

## Paises seleccionados (5)

| #   | Pais            | Codigo ISO | Por que lo elegiste                                                            |
| --- | --------------- | ---------- | ------------------------------------------------------------------------------ |
| 1   | **Afghanistan** | AFG        | Actor central histórico y geopolítico en la región ("Cementerio de Imperios"). |
| 2   | **Mongolia**    | MNG        | Estado tapón estratégico y neutral entre las potencias Rusia y China.          |
| 3   | **Azerbaijan**  | AZE        | Pieza clave en la conexión Caspio-Cáucaso y seguridad energética.              |
| 4   | **Georgia**     | GEO        | Referente de aspiraciones democráticas y occidentales en la región.            |
| 5   | **Armenia**     | ARM        | Aliado estratégico tradicional de Rusia en el Cáucaso Sur.                     |

**IMPORTANTE:** No puedes usar los paises del ejemplo del profesor (KAZ, UZB, TKM, KGZ, TJK).

---

## Variables seleccionadas (5 numericas)

| #   | Variable QoG | Que mide              | Por que la elegiste                                                            |
| --- | ------------ | --------------------- | ------------------------------------------------------------------------------ |
| 1   | `gle_cgdpc`  | PIB per cápita real   | **Variable Objetivo (Target):** Indicador estándar de desarrollo económico.    |
| 2   | `wdi_expmil` | Gasto militar (% PIB) | **Poder Duro:** Refleja la priorización de seguridad sobre bienestar.          |
| 3   | `p_polity2`  | Índice de Democracia  | **Poder Blando:** Mide la estabilidad y apertura del régimen político.         |
| 4   | `vdem_corr`  | Índice de Corrupción  | **Calidad Institucional:** Factor clave que afecta la inversión y crecimiento. |
| 5   | `wdi_lifexp` | Esperanza de vida     | **Variable de Control:** Indicador básico de bienestar social y salud.         |

**Tip:** Consulta el codebook de QoG para entender que mide cada variable:
<https://www.gu.se/en/quality-government/qog-data>

---

## Variable derivada

He creado la variable **`subregion`** para agrupar geográficamente a los países y capturar dinámicas regionales distintas más allá de las fronteras nacionales:

- **Caucasus:** Azerbaiyán, Georgia, Armenia.
- **Central/South:** Afganistán.
- **East:** Mongolia.

---

## Tipo de analisis elegido

- [ ] Clustering (K-Means)
- [ ] Serie temporal (evolucion por pais)
- [x] Comparacion (Regresión Random Forest - Importancia de Factores)
- [x] Modelo Econométrico (Panel Data - Efectos Fijos vs Aleatorios)

---

## Como ejecutar mi pipeline

```bash
# Paso 1: Levantar infraestructura
docker compose up -d

# Paso 2: Verificar que todo funciona
docker ps

# Paso 3: Descargar Datos (Automático)
docker exec jupyter_lab python /home/jovyan/work/src/download_data.py

# Paso 4: Ejecutar pipeline ETL (Procesamiento de Datos)
# Nota: Ejecutar desde dentro del contenedor o tener Spark local.
# Si usas Docker (recomendado):
docker exec jupyter_lab python /home/jovyan/work/src/pipeline.py

# Paso 5: Ejecutar Análisis y Generar Gráficos
docker exec jupyter_lab spark-submit /home/jovyan/work/src/analysis.py

# Paso 6: Ejecutar Análisis Econométrico (Hausman)
# Nota: Ejecutar desde entorno con librerías 'linearmodels' instaladas (puede ser local si tienes entorno)
docker exec jupyter_lab python /home/jovyan/work/src/econometric_analysis.py
# (Si no corre en docker por falta de librerías, instalar: pip install linearmodels statsmodels)
```

El análisis generará los gráficos en la carpeta `notebooks/` y el reporte final está en `03_RESULTADOS.md`.

---

## Estructura del Proyecto

```text
├── .github/workflows/          # [DevOps] Automatización de despliegue web
├── docs/                       # [Doc] Documentación oficial (Markdown)
│   ├── 01_README.md            # Este archivo (Guía de Trabajo)
│   ├── 02_INFRAESTRUCTURA.md   # Detalles de Docker
│   ├── 03_RESULTADOS.md        # Informe de hallazgos
│   ├── 05_EXPLICACION_CODIGO.md# Catálogo técnico de scripts
│   └── ...                     # Otros archivos de documentación
├── src/                        # [Code] Código Fuente Python
│   ├── pipeline.py             # Lógica ETL Big Data
│   ├── analysis.py             # ML Engine (Random Forest)
│   ├── econometric_analysis.py # Stats Engine (Hausman)
│   ├── app_streamlit.py        # Dashboard (Versión Clásica)
│   └── app_streamlit_pro.py    # Dashboard (Versión PRO 3D)
├── data/                       # [Data] Lakehouse (Raw + Processed)
├── docker-compose.yml          # [Infra] Orquestación de contenedores
├── mkdocs.yml                  # [Config] Configuración de la web de docs
└── upload_to_github.bat        # [Tool] Script de subida automática
```

---

## 📄 Licencia

Este proyecto está bajo la Licencia **MIT**. Consulta el archivo [LICENSE](../LICENSE) para más detalles.

Copyright (c) 2026 **Alexis M.**
