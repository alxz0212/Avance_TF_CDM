# Paso 6: Preguntas de Comprensión

**Alumno:** Daniel Alexis Mendoza Corne

> **Instrucciones:** Responde cada pregunta con tus propias palabras.
> Las respuestas deben ser específicas y demostrar que entiendes los conceptos.
> Se acepta entre 3-5 oraciones por pregunta.
>
> **Nota:** Completa este archivo AL FINAL, después de haber terminado
> los bloques A, B y C. Así tendrás la experiencia necesaria para responder.

---

## 1. Infraestructura

**Si tu worker tiene 2 GB de RAM y el CSV pesa 3 GB, ¿qué pasa?
¿Cómo lo solucionarías?**

Si intento cargar todo el archivo de golpe (como hace Pandas), me daría un error de memoria (`Out Of Memory`) porque 3GB no entran en 2GB. Pero Spark es inteligente: no carga todo, sino que divide el archivo en "pedacitos" (particiones) y los procesa uno por uno. Si aun así fallara, la solución sería aumentar el número de particiones con `.repartition()` para que cada trozo sea más pequeño, o añadir más nodos Worker al cluster para repartir la carga.

---

## 2. ETL

**¿Por qué `spark.read.csv()` no ejecuta nada hasta que llamas
`.count()` o `.show()`?**

Porque Spark es "perezoso" (Lazy Evaluation). Cuando le digo "lee esto" o "filtra aquello", en realidad no lo hace al momento, solo se apunta la tarea en un plan (DAG). Solo se pone a trabajar de verdad cuando le pido un resultado final (una acción como `count` o `show`). Esto es genial porque si filtro datos, Spark se da cuenta y ni siquiera lee lo que no necesito, ahorrando tiempo.

---

## 3. Análisis

**Interpreta tu gráfico principal: ¿qué patrón ves y por qué crees
que ocurre?**

Mirando el gráfico de importancia de variables (Random Forest), veo que el **Gasto Militar** influye mucho más que la democracia. Esto tiene sentido para mi zona de estudio ("El Gran Juego"): parece que los países que más crecen económicamente (como Azerbaiyán) no son los más democráticos, sino los que tienen un estado fuerte y militarizado para mantener el control. Vamos, que la estabilidad política pesa más que la libertad en esta región.

![Feature Importance](notebooks/grafico_feature_importance.png)

> **Leyenda de Variables:**
>
> - `wdi_lifexp`: Esperanza de Vida (Salud/Social)
> - `wdi_expmil`: Gasto Militar (Poder Duro)
> - `vdem_corr`: Control de Corrupción (Institucional)
> - `p_polity2`: Índice de Democracia (Poder Blando)

---

## 4. Escalabilidad

**Si tuvieras que repetir este ejercicio con un dataset de 50 GB,
¿qué cambiarías en tu infraestructura?**

¡Mi portátil explotaría! Docker Desktop no aguantaría eso. Tendría que llevarme el proyecto a la nube (como AWS o Databricks) y usar un sistema de almacenamiento distribuido de verdad (HDFS o S3) en vez de mi disco duro. Además, necesitaría un cluster con varios Workers (máquinas conectadas), porque un solo nodo no podría con tanto volumen de datos en un tiempo razonable.
