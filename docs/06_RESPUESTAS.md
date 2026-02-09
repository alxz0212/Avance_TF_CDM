# Paso 6: Preguntas de Comprension

**Alumno:** Daniel Alexis Mendoza Corne

> **Instrucciones:** Responde cada pregunta con tus propias palabras.
> Las respuestas deben ser especificas y demostrar que entiendes los conceptos.
> Se acepta entre 3-5 oraciones por pregunta.
>
> **Nota:** Completa este archivo AL FINAL, despues de haber terminado
> los bloques A, B y C. Asi tendras la experiencia necesaria para responder.

---

## 1. Infraestructura

**Si tu worker tiene 2 GB de RAM y el CSV pesa 3 GB, que pasa?
Como lo solucionarias?**

Si intento cargar todo el archivo en memoria de una sola vez (como haría Pandas), el proceso fallaría con un error de `Out Of Memory (OOM)`. Sin embargo, Spark está diseñado para manejar esto: dividiría el archivo en particiones más pequeñas y procesaría "trozos" del archivo uno por uno (pipelining), o haría "spill to disk" (mermando rendimiento pero sin crashear). Si el error persiste, la solución óptima es aumentar el número de particiones (`df.repartition(n)`) para que cada trozo quepa cómodamente en los 2GB, o escalar horizontalmente agregando otro nodo Worker.

---

## 2. ETL

**Por que `spark.read.csv()` no ejecuta nada hasta que llamas
`.count()` o `.show()`?**

Esto se debe a la **"Lazy Evaluation"** (Evaluación Perezosa) de Spark. Cuando ejecuto `read` o `filter`, Spark no procesa datos inmediatamente, sino que solo "toma nota" de qué hacer, construyendo un plan de ejecución optimizado (DAG). Solo cuando le pido un resultado final (una "Acción" como `count` o `show`), Spark dispara realmente el procesamiento. Esto es vital para optimizar: si filtro y luego selecciono una columna, Spark sabe que no necesita leer las columnas que no voy a usar.

---

## 3. Analisis

**Interpreta tu grafico principal: que patron ves y por que crees
que ocurre?**

En mi gráfico de Feature Importance (Random Forest), observo que, aislando la **Esperanza de Vida (`wdi_lifexp`)** (que es obvia), el **Gasto Militar (`wdi_expmil`)** tiene un peso predictivo superior a las variables puramente democráticas. Esto sugiere un patrón geopolítico en la región ("El Gran Juego"): los estados que han logrado mayor desarrollo económico relativo (como Azerbaiyán o Kazajistán) no son necesariamente los más democráticos, sino aquellos con estados fuertes y militarizados capaces de mantener estabilidad en una zona de conflicto.

![Feature Importance](notebooks/grafico_feature_importance.png)

> **Leyenda de Variables:**
> *   `wdi_lifexp`: Esperanza de Vida (Salud/Social)
> *   `wdi_expmil`: Gasto Militar (Poder Duro)
> *   `vdem_corr`: Control de Corrupción (Institucional)
> *   `p_polity2`: Índice de Democracia (Poder Blando)

---

## 4. Escalabilidad

**Si tuvieras que repetir este ejercicio con un dataset de 50 GB,
que cambiarias en tu infraestructura?**

Primero, mi laptop y Docker Desktop no aguantarían; movería la infraestructura a la nube (AWS EMR o Databricks). Cambiaría el almacenamiento local por un sistema distribuido real como **HDFS o S3 (Data Lake)**, ya que mi disco local sería un cuello de botella. Finalmente, en lugar de 1 solo contenedor Worker, configuraría un cluster con múltiples nodos (ej. 5 o 10 Workers) para aprovechar el paralelismo real, asegurando que cada nodo procese solo unos pocos GBs de esos 50GB totales.
