# Paso 4: Reflexión IA - "3 Momentos Clave"

**Alumno:** Daniel Alexis Mendoza Corne  
**Fecha:** Febrero 2026

---

## Bloque A: Infraestructura (Docker)

### 1. Arranque

**¿Qué fue lo primero que le pediste a la IA?**  
Le pedí ayuda para crear el archivo `docker-compose.yml`. No tenía muy claro cómo conectar Spark (el Master y el Worker) con JupyterLab y la base de datos Postgres, así que le pedí que me generara la estructura básica para que todo funcionara junto.

### 2. Error

**¿Qué falló y cómo lo resolviste?**  
Al principio no podía entrar a los servicios. Intentaba poner en el navegador los puertos que veía en el archivo, como el `7077` o el `5432`, y me salía error de página no encontrada.

- **Resolución:** La IA me explicó que esos puertos son internos para que se hablen las máquinas entre ellas. Para yo ver algo, tenía que usar los puertos "web" o visuales, que eran el `8080` para ver Spark y el `8888` para mi Jupyter. ¡Vaya lío de puertos!

**Otro problema: PySpark no aparecía**

- **Fallo:** Cuando corría mi código, me decía `ModuleNotFoundError: No module named 'pyspark'`.
- **Solución:** Resulta que aunque la imagen de Docker tenía Spark, mi script de Python no lo encontraba. Tuve que añadir `pyspark==3.5.0` al archivo `requirements.txt` y volver a construir la imagen.

### 3. Aprendizaje

**¿Qué aprendiste que NO sabías antes?**  
Entendí la diferencia entre los puertos que "expongo" hacia afuera (para mí) y los que se quedan dentro de la red de Docker. También aprendí a usar `volumes` para guardar mis notebooks, porque la primera vez reinicié el contenedor y... ¡adiós trabajo!

**Otro Error Detectado: Spark Worker Offline**

- **Fallo:** En la interfaz `localhost:8080`, aparecía "Alive Workers: 0" aunque el contenedor existía.
- **Causa:** Al reconstruir y levantar solo el servicio `jupyter-lab`, docker-compose no necesariamente reinicia o mantiene activos los contenedores dependientes si no se especifican.
- **Resolución:** Ejecutar `docker-compose up -d` (sin especificar servicio) y verificar con `docker ps` aseguró que tanto Master como Worker estuvieran activos.
- **Aprendizaje:** La "Arquitectura Distribuida" requiere validación explícita de que todos los nodos están vivos, no basta con que el código corra (que puede estar en modo local).

### 💬 Prompt Clave (Bloque A)

```text
"Genera la configuración de docker-compose.yml para un entorno Spark clusterizado (1 Master, 1 Worker) con persistencia de datos en PostgreSQL. Asegúrate de exponer los puertos UI (8080, 4040) y configurar la red bridge para que JupyterLab pueda acceder al Spark Master mediante el nombre del servicio."
```

---

## Bloque B: Pipeline ETL (Spark)

### 1. Arranque

**¿Qué fue lo primero que le pediste a la IA?**  
Necesitaba ayuda para hacer el script `pipeline.py`. Quería que leyera el dataset QoG pero que solo se quedara con los 5 países que me interesaban del "Gran Juego" y que además me creara una columna nueva para agruparlos por zona.

### 2. Error

**¿Qué falló y cómo lo resolviste?**  
Tuve problemas graves al intentar subir todo a GitHub.
**Error:** `fatal: not a git repository`.

- **Resolución:** Me había olvidado de iniciar el repositorio con `git init`. La IA me guio paso a paso: iniciar git, configurar el `.gitignore` (súper importante para no subir datos pesados por error) y luego vincularlo con mi repo en GitHub.

### 3. Aprendizaje

**¿Qué aprendiste que NO sabías antes?**  
Aprendí a usar `pyspark.sql.functions.when`. Antes hacía esto con bucles `for` en Python normal, pero con Big Data eso es lentísimo. Con esta función de Spark, puedo crear columnas condicionales (como la de `subregion`) de forma súper rápida y distribuida.

### 💬 Prompt Clave (Bloque B)

```text
"Quiero subir mi proyecto a GitHub pero evitar errores. Dame los pasos exactos para iniciar el repositorio, crear un archivo `.gitignore` que excluya mis datos pesados (carpeta /data) y los archivos temporales de Jupyter, y finalmente cómo conectar mi carpeta local con el repositorio remoto main."
```

---

## Bloque C: Análisis de Datos (Machine Learning)

### 1. Arranque

**¿Qué fue lo primero que le pediste a la IA?**  
Le pregunté qué modelo de Machine Learning me convenía más. Estaba dudando entre KNN, SVM o Random Forest para ver cómo influían las variables políticas en la economía.

### 2. Error

**¿Qué falló y cómo lo resolviste?**  
Quise generar las gráficas automáticamente corriendo el notebook desde la terminal, pero todo explotó.
**Error:** `TypeError: 'JavaPackage' object is not callable`.

- **Resolución:** La IA me sugirió que no mezclara cosas. En lugar de forzar el notebook, pasé la lógica a un script limpio en Python (`src/analysis.py`) y lo ejecuté con `spark-submit`. Funcionó mucho mejor y sin conflictos raros de Java.

### 3. Aprendizaje

**¿Qué aprendiste que NO sabías antes?**  
Que **Random Forest** es genial no solo para predecir, sino para explicar **por qué** predice lo que predice (feature importance). Eso me ayudó mucho más que KNN para entender mi problema de investigación. También aprendí que automatizar gráficas es mejor que hacerlas a mano una por una.

### 💬 Prompt Clave (Bloque C)

```text
"Actúa como un experto en Data Science. Tengo un dataset con variables sociopolíticas y quiero predecir el impacto en el PIB. Evalúa comparativamente KNN, SVM y Random Forest justificando cuál es más adecuado considerando la explicabilidad (feature importance), manejo de outliers y la dimensionalidad de mis datos."
```
