# Plan de Implementación - Trabajo Final Big Data & Docker

Este documento guiará el desarrollo del proyecto etapa por etapa, comenzando por la infraestructura.

## Etapa 1: Infraestructura Docker

### Objetivo

Crear un entorno contenerizado que soporte **JupyterLab**, **Apache Spark** (PySpark) y **TensorFlow**.

### Revisión de Usuario Requerida

> [!IMPORTANT]
> Por favor conforma si la imagen base debe ser alguna en específico o si creamos una imagen custom `jupyter/pyspark-notebook` e instalamos TensorFlow encima.

### Cambios Propuestos

#### Archivos de Configuración [NUEVO]

1.  **`docker/Dockerfile`**:
    - Base: `jupyter/pyspark-notebook` (Recomendada para Big Data + Python).
    - Instalación adicional: `tensorflow`, `pandas`, `matplotlib`.

2.  **`docker-compose.yml`** (En la raíz o carpeta `docker/`):
    - Servicio `spark-tf-lab`:
      - Puerto: `8888` (Jupyter)
      - Volúmenes: `./data:/home/jovyan/work/data`, `./notebooks:/home/jovyan/work/notebooks`, `./src:/home/jovyan/work/src`
    - Variables de entorno para configurar Spark localmente.

### Verificación de la Etapa 1

1.  Ejecutar `docker-compose up --build`.
2.  Acceder a JupyterLab.
3.  Ejecutar `import pyspark` y `import tensorflow` en un notebook de prueba.

## Etapa 2: Configuración de Dependencias

### Objetivo

Definir y congelar las versiones de las librerías necesarias para el proyecto (Spark, Pandas, Visualización, Drivers de BBDD) en un archivo estándar `requirements.txt`.

### Cambios Propuestos

#### Raíz del Proyecto [NUEVO]

1.  **`requirements.txt`**:
    - `pyspark`
    - `pandas`
    - `matplotlib`
    - `seaborn`
    - `sqlalchemy`
    - `psycopg2-binary`
    - `jupyterlab`
    - `plotly`
    - `scikit-learn`
    - `graphviz`

### Verificación de la Etapa 2

1.  Verificar la existencia del archivo `requirements.txt`.
2.  (Opcional) Instalar dependencias en entorno local: `pip install -r requirements.txt`.

## Etapa 3: Pipeline ETL con Spark (Bloque B)

### Objetivo

Desarrollar un script `pipeline.py` que procese el dataset Quality of Government (QoG) utilizando Apache Spark, enfocándose en la **Periferia de Asia Central y el "Gran Juego"**.

### Selección de Datos

#### Países (5)

Selección basada en el contexto del "Gran Juego" y la influencia post-soviética (excluyendo KAZ, UZB, TKM, KGZ, TJK):

1.  **Afghanistan** (AFG) - Actor central histórico y geopolítico.
2.  **Mongolia** (MNG) - Estado tapón estratégico entre Rusia y China.
3.  **Azerbaijan** (AZE) - Conexión Caspio-Cáucaso, clave en energía.
4.  **Georgia** (GEO) - Referente de aspiraciones occidentales en la región post-soviética.
5.  **Armenia** (ARM) - Aliado estratégico tradicional de Rusia en el Cáucaso.

#### Variables (5)

1.  `gle_cgdpc` (PIB per cápita real) - Económica.
2.  `wdi_lifexp` (Esperanza de vida) - Social/Desarrollo.
3.  `p_polity2` (Índice de Democracia - Polity IV) - Política (Régimen).
4.  `vdem_corr` (Índice de corrupción V-Dem) - Política (Institucional). _Alternativa: `ti_cpi` si vdem tiene muchos nulos_.
5.  `wdi_mil` (Gasto militar % PIB) - Geopolítica/Seguridad (si disponible, sino `wdi_pop`).

#### Pregunta de Investigación

> "¿Cómo ha influido la estabilidad democrática (o falta de ella) en el desarrollo económico y el gasto militar de los estados 'amortiguadores' de Asia Central (Afganistán, Mongolia, Cáucaso) tras la caída de la URSS?"

### Cambios Propuestos

#### Raíz del Proyecto [NUEVO]

1.  **`src/pipeline.py`**:
    - **Inicialización**: Crear `SparkSession`.
    - **Ingesta**: Leer `data/raw/qog_std_ts_jan26.csv`.
    - **Filtrado**:
      - `cname` en [Afghanistan, Mongolia, Azerbaijan, Georgia, Armenia]
      - `year` >= 1991 (Disolución URSS) hasta actualidad.
    - **Selección**: Variables definidas.
    - **Transformación**:
      - Crear col `subregion`: 'Caucasus' (AZE, GEO, ARM), 'Central/South' (AFG), 'East' (MNG).
      - Generar métricas derivadas (ej. crecimiento PIB vs Gasto Militar).
    - **Carga**: Guardar como `data/processed/qog_great_game.parquet`.

### Verificación de la Etapa 3

1.  Ejecutar el script: `python src/pipeline.py`.
2.  Verificar la generación de `data/processed/qog_great_game.parquet`.

## Etapa 4: Análisis Avanzado y ML (Bloque C)

### Objetivo

Responder la pregunta de investigación mediante visualización de datos y un modelo de Machine Learning (**Random Forest**) para determinar la importancia de los factores geopolíticos vs democráticos en la economía.

### Pregunta Refinada

> "¿Son los factores de 'Poder Duro' (Gasto Militar) o de 'Poder Blando' (Democracia, Control de Corrupción) los que determinan el desarrollo económico en la periferia post-soviética?"

### Metodología

1.  **Carga de Datos**: Leer `data/processed/qog_great_game.parquet` con PySpark.
2.  **Preparación ML**:
    - `VectorAssembler` para unificar features: `wdi_expmil`, `p_polity2`, `vdem_corr`, `wdi_lifexp`.
    - Target: `gle_cgdpc`.
3.  **Modelado**: Entrenar `RandomForestRegressor`.
4.  **Interpretación**: Extraer `featureImportances` y visualizar qué variable tiene mayor peso.

### Verificación

1.  Ejecutar el notebook `notebooks/02_analisis_gran_juego.ipynb`.
2.  Obtener el gráfico de Importancia de Variables.
