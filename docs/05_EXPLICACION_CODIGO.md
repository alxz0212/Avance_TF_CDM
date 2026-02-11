# 📘Paso 5: Documentación Técnica del Código Fuente

**Proyecto:** Big Data & Geopolítica ("El Gran Juego")
**Alumno:** Daniel Alexis Mendoza Corne
**Fecha:** Febrero 2026

---

## 1. ¿Por qué la carpeta se llama `src`?

`src` es la abreviatura estándar en ingeniería de software para **"Source"** (Código Fuente).

En proyectos profesionales, es fundamental mantener separado el código lógico de otros elementos. Esta estructura garantiza:

- **Orden:** El código no se mezcla con la documentación (`.md`), la configuración (`docker/`) o los datos (`data/`).
- **Seguridad:** Facilita la configuración de permisos; por ejemplo, el servidor de producción solo necesita acceso de lectura a `src`, pero de escritura a `data`.
- **Escalabilidad:** Si el proyecto crece, todo el código lógica reside en un único punto de verdad.

---

## 2. Catálogo de Scripts

A continuación, se detalla la función técnica y de negocio de cada módulo desarrollado.

### 🛠️ 1. Infraestructura y Preparación

#### `download_data.py`

- **Función:** Automatización de Ingesta.
- **Qué hace:** Se conecta al repositorio de la Universidad de Gotemburgo, descarga el dataset `.csv` de 68MB y lo coloca en la ruta `data/raw/`.
- **Por qué es importante:** Elimina la dependencia de descargas manuales, haciendo que el proyecto sea reproducible en cualquier máquina con un solo comando.

#### `verify_spark.py`

- **Función:** Test de Integridad (Smoke Test).
- **Qué hace:** Intenta iniciar una sesión de Spark y crear un DataFrame pequeño en memoria.
- **Por qué es importante:** Es el primer script que ejecutamos para validar que el contenedor de Docker y el cluster de Spark están comunicándose correctamente antes de lanzar procesos pesados.

---

### ⚙️ 2. Procesamiento de Datos (ETL)

#### `pipeline.py`

- **Función:** ETL (Extract, Transform, Load).
- **Tecnología:** Apache Spark (PySpark SQL).
- **Flujo de Trabajo:**
  1. **Extract:** Lee el CSV crudo.
  2. **Transform:**
      - Filtra los 5 países del "Gran Juego" (Afganistán, Mongolia, Cáucaso).
      - Crea la variable derivada `subregion`.
      - Castea tipos de datos (Strings a Doubles) para asegurar precisión matemática.
  3. **Load:** Guarda el resultado limpio en formato **Parquet**.
- **Detalle Pro:** Usamos `.parquet` en lugar de `.csv` porque es un formato columnar comprimido que es mucho más rápido para leer en análisis posteriores de Big Data.

#### `ingest_data.py` (Módulo Legado)

- **Función:** Conector a Base de Datos Relacional.
- **Qué hace:** Estaba diseñado para cargar los datos en PostgreSQL.
- **Estado:** Se mantiene como respaldo. Para el análisis principal optamos por el flujo Spark-Parquet por ser más nativo del ecosistema de Big Data que el almacenamiento SQL tradicional.

---

### 🧠 3. Análisis Avanzado y Resultados

#### `analysis.py`

- **Función:** Motor de Machine Learning.
- **Tecnología:** Spark MLlib.
- **Qué hace:**
  - Carga los datos procesados (Parquet).
  - **Matriz de Correlación:** Calcula cómo se relacionan las variables (ej. Gasto Militar vs PIB).
  - **Random Forest:** Entrena un modelo de Inteligencia Artificial compuesto por 100 árboles de decisión para predecir el desarrollo económico.
  - **Feature Importance:** Extrae qué variables tuvieron más peso en la decisión del modelo.
- **Salida:** Genera automáticamente los gráficos estáticos `.png` en la carpeta `notebooks/`.

#### `econometric_analysis.py`

- **Función:** Análisis Econométrico Riguroso.
- **Tecnología:** Librería `linearmodels` (Python).
- **Qué hace:**
  - Ejecuta modelos de regresión para datos de panel: **Efectos Fijos (Fixed Effects)** y **Efectos Aleatorios (Random Effects)**.
  - Implementa el **Test de Hausman** para determinar cuál de los dos modelos es estadísticamente más adecuado (causalidad vs correlación).
  - Genera un reporte detallado en `notebooks/hausman_results.txt`.
- **Valor agregado:** Complementa la "caja negra" del Machine Learning (Random Forest) con inferencia estadística clásica, validando si las características únicas de cada país sesgan los resultados.

---

### 🚀 4. Interfaz de Usuario (Frontend)

#### `src/app_streamlit.py` y `src/app_streamlit_pro.py`

Son el Frontend de la aplicación.

- **Tecnología:** Streamlit.
- **Funciones:**
  - Cargar el Parquet procesado.
  - Generar gráficos interactivos con Plotly.
  - **Pro Version:** Incluye globo 3D, radar charts y estética "Dark Mode".
  - Sirve una interfaz web en el puerto `8501`.
  - Permite al usuario explorar los datos: filtrar por año, ver tendencias temporales interactivas y simular predicciones.
  - Es la "cara" del proyecto, transformando el código técnico en un producto visual consumible por un usuario final.

---

## 3. Diagrama de Flujo de Datos

```mermaid
graph TD
    %% Estilos
    classDef source fill:#f9f,stroke:#333,stroke-width:2px;
    classDef script fill:#bbf,stroke:#333,stroke-width:2px,color:black;
    classDef data fill:#dfd,stroke:#333,stroke-width:2px,color:black;
    classDef output fill:#fd9,stroke:#333,stroke-width:2px,color:black,stroke-dasharray: 5 5;

    subgraph INGESTA ["📡 Ingesta de Datos"]
        A["☁️ Internet / Repo QoG"]:::source
        Script1{{"🐍 download_data.py"}}:::script
    end

    subgraph PROCESAMIENTO ["⚙️ Procesamiento & Análisis"]
        Script2{{"⚡ pipeline.py"}}:::script
        Script3{{"🧠 analysis.py"}}:::script
        Script5{{"📉 econometric_analysis.py"}}:::script
    end

    subgraph ALMACENAMIENTO ["💾 Almacenamiento"]
        B[("📄 Raw CSV")]:::data
        C[("📦 Clean Parquet")]:::data
    end

    subgraph VISUALIZACION ["📊 Consumo & UI"]
        Script4{{"🚀 app_streamlit_pro.py"}}:::script
        D["📈 Gráficos Estáticos .png"]:::output
        E["🖥️ Dashboard 3D Interactivo"]:::output
        F["📄 Reporte Hausman .txt"]:::output
    end

    %% Relaciones
    A --> Script1
    Script1 --> B
    B --> Script2
    Script2 --> C
    C --> Script3
    C --> Script4
    C --> Script5
    Script3 --> D
    Script4 --> E
    Script5 --> F
```

> [!NOTE]
> **Conclusión del Flujo de Datos:**  
> Como se observa en el diagrama, el proyecto sigue una arquitectura lineal de Big Data moderna:
>
> 1. **Ingesta:** Los datos se capturan automáticamente de internet (`download_data.py`).
> 2. **Procesamiento:** Se limpian y estructuran en Spark (`pipeline.py`), guardándose en formato eficiente **Parquet**.
> 3. **Consumo:** A partir del dato limpio, se derivan tres productos finales: Análisis ML (`analysis.py`), Validación Estadística (`econometric_analysis.py`) y Visualización Interactiva (`app_streamlit_pro.py`).
>
> Esta estructura modular asegura que si cambiamos la fuente de datos, solo tocamos el script de _Ingesta_, sin romper el Dashboard final.

---

## 4. DevOps y Documentación (Automatización) 🏗️

Para desplegar este sitio web, utilizamos un flujo de trabajo automatizado que se basa en el formato **.yml** (YAML).

### ¿Qué es un archivo .yml / .yaml?

Es un formato de "serialización de datos" diseñado para ser leído fácilmente por humanos.

- **Configuración:** Se usa casi universalmente en DevOps para configurar herramientas.
- **Indentación:** Se basa estrictamente en espacios (niveles). ¡Un espacio de más o de menos puede invalidar el archivo!
- **Estructura:** Funciona mediante pares de `clave: valor`.

### Relación entre los elementos clave

#### 1. `mkdocs.yml` (El Cerebro 🧠)

- **Ubicación:** Raíz del proyecto.
- **Función:** Es el archivo central de configuración de tu sitio web.
- **Navegación & index.md:** Aquí es donde asocias tus archivos Markdown con el menú de la web. Por ejemplo:
  - `- "🏠 Inicio": index.md`
  - Esto le dice a MkDocs que cuando alguien haga clic en "Inicio", debe mostrar el contenido de `index.md`.
- **Estética:** Aquí defines si el sitio es oscuro (Slate) o claro, los colores primarios y los iconos.

#### 2. `.github/workflows/deploy_docs.yml` (El Robot 👷)

- **Ubicación:** `.github/workflows/`.
- **Función:** Automatización del Despliegue (CI/CD).
- **Flujo:** Cada vez que haces un `git push` a la rama `main`, este "robot" se despierta y:
  1. Lee el `mkdocs.yml` para entender la estructura.
  2. Procesa el `index.md` y los demás archivos `.md`.
  3. Transforma todo en código HTML real (el sitio web).
  4. Lo publica en GitHub Pages.

#### 3. `index.md` (La Puerta de Entrada 🚪)

- Es el archivo de contenido más importante. Es la "Home" de tu documentación.
- Sin un archivo designado como inicio en el `mkdocs.yml`, la web no tendría una página de aterrizaje clara.
