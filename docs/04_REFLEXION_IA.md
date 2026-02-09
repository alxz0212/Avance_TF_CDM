# Paso 4: Reflexión IA - "3 Momentos Clave"

**Alumno:** Daniel Alexis Mendoza Corne  
**Fecha:** Febrero 2026

---

## Bloque A: Infraestructura (Docker)

### 1. Arranque

**¿Qué fue lo primero que le pediste a la IA?**  
Le pedí generar un archivo `docker-compose.yml` que incluyera servicios para Spark (Master y Worker), PostgreSQL como base de datos y JupyterLab como entorno de desarrollo interactivo, asegurando la conexión entre ellos.

### 2. Error

**¿Qué falló y cómo lo resolviste?**  
Al intentar verificar los servicios, intenté acceder vía navegador a los puertos `7077` (Spark Master) y `5432` (Postgres) obteniendo una página de error ("Empty Response").

- **Resolución:** Aprendí que esos son puertos de comunicación interna (TCP) para los servicios, no interfaces web HTTP. Me dirigí a los puertos correctos visuales: `8080` (Spark UI) y `8888` (JupyterLab).

**Otro Error Detectado: PySpark Module**
- **Fallo:** `ModuleNotFoundError: No module named 'pyspark'` al correr scripts internos.
- **Causa:** La imagen base contiene Spark pero no el paquete pip accesible por defecto en scripts externos.
- **Resolución:** Se añadió explícitamente `pyspark==3.5.0` en `requirements.txt` para hacer match con la versión binaria del contenedor.

### 3. Aprendizaje

**¿Qué aprendiste que NO sabías antes?**  
La diferencia crítica entre los puertos expuestos para clientes (Navegador) y los puertos de servicio interno en Docker. También cómo persistir datos usando `volumes` para no perder mis notebooks al reiniciar el contenedor.

**Otro Error Detectado: Spark Worker Offline**
- **Fallo:** En la interfaz `localhost:8080`, aparecía "Alive Workers: 0" aunque el contenedor existía.
- **Causa:** Al reconstruir y levantar solo el servicio `jupyter-lab`, docker-compose no necesariamente reinicia o mantiene activos los contenedores dependientes si no se especifican.
- **Resolución:** Ejecutar `docker-compose up -d` (sin especificar servicio) y verificar con `docker ps` aseguró que tanto Master como Worker estuvieran activos.
- **Aprendizaje:** La "Arquitectura Distribuida" requiere validación explícita de que todos los nodos están vivos, no basta con que el código corra (que puede estar en modo local).

### 💬 Prompt Clave (Bloque A)

```text
"enrtonces ya esta actualizado 02_INFRAESTRUCTURA..md y docker-compose.yml para verificar?"
```

---

## Bloque B: Pipeline ETL (Spark)

### 1. Arranque

**¿Qué fue lo primero que le pediste a la IA?**  
Cómo estructurar un script `pipeline.py` que leyera el dataset QoG, filtrara específicamente los 5 países seleccionados de mi zona de estudio ("Gran Juego") y creara una variable derivada para agruparlos por subregión.

### 2. Error

**¿Qué falló y cómo lo resolviste?**  
Tuve dificultades iniciales con el control de versiones al intentar subir el proyecto.
**Error:** `fatal: not a git repository`.

- **Resolución:** Inicialicé correctamente el repositorio con `git init`, configuré el `.gitignore` protegiendo la carpeta `data/` y añadí el remoto de GitHub correctamente antes de hacer el push.

### 3. Aprendizaje

**¿Qué aprendiste que NO sabías antes?**  
Cómo utilizar `pyspark.sql.functions.when` para crear columnas condicionales complejas (variable `subregion`) de manera eficiente en un DataFrame distribuido, evitando bucles `for` de Python que son ineficientes en Big Data.

### 💬 Prompt Clave (Bloque B)

```text
"sube todo a mi github https://github.com/alxz0212/Avance_TF_CDM"
```

---

## Bloque C: Análisis de Datos (Machine Learning)

### 1. Arranque

**¿Qué fue lo primero que le pediste a la IA?**  
Solicité asesoramiento para elegir el mejor modelo de Machine Learning (entre KNN, SVM y Random Forest) dado mi objetivo de explicar la influencia de factores políticos en la economía.

### 2. Error

**¿Qué falló y cómo lo resolviste?**  
Al intentar generar los gráficos automáticamente ejecutando el notebook desde la terminal con `nbconvert`.
**Error:**

```text
TypeError: 'JavaPackage' object is not callable
...
spark = SparkSession.builder...
```

- **Resolución:** El entorno de ejecución automática tenía conflictos con la sesión de Spark existente. Migré la lógica a un script python dedicado (`src/analysis.py`) ejecutado con `spark-submit`, lo cual demostró ser mucho más robusto para tareas de producción.

### 3. Aprendizaje

**¿Qué aprendiste que NO sabías antes?**  
Que `Random Forest` no solo predice, sino que ofrece la métrica `featureImportances` que sirve para explicar causalidad ("Explicabilidad del modelo"), haciéndolo superior a KNN para mi pregunta de investigación. También aprendí a automatizar la generación de gráficos sin abrir Jupyter manualmente.

### 💬 Prompt Clave (Bloque C)

```text
"Antes de empezar quisiera aplicar dentro del analisis uno de estos modelos KNN , SVM o Ramdom forest cual crees que seria mejor teniendo en cuenta la data que tengo"
```
