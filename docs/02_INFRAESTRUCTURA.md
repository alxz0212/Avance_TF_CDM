# Paso 2: Infraestructura Docker

- **Alumno:** Daniel Alexis Mendoza Corne
- **Fecha:** Febrero 2026

---

## 2.1 Mi docker-compose.yml explicado

### Servicio: PostgreSQL

```yaml
postgres:
  image: postgres:13
  container_name: postgres_db
  environment:
    POSTGRES_USER: user
    POSTGRES_PASSWORD: password
    POSTGRES_DB: qog_data
  ports:
    - "5432:5432"
  volumes:
    - ./data/postgres:/var/lib/postgresql/data
  networks:
    - bigdata-net
```

**Que hace:**
Este servicio levanta una base de datos relacional PostgreSQL versión 13.

- `container_name`: Le da un nombre fijo (`postgres_db`) para que otros contenedores (como Spark o Jupyter) puedan encontrarlo fácilmente en la red interna usando ese nombre como si fuera un dominio DNS.
- `environment`: Define las credenciales (usuario/contraseña) y el nombre de la base de datos que se creará automáticamente al iniciar.
- `ports`: Mapea el puerto 5432 del contenedor al 5432 de mi máquina local ("host"), permitiéndome conectarme desde fuera de Docker (por ejemplo, con DBeaver).
- `volumes`: Crea una persistencia de datos. Si borro el contenedor, los datos guardados en `/var/lib/postgresql/data` (dentro del contenedor) se conservan en mi carpeta local `./data/postgres`.
- `networks`: Conecta el servicio a una red compartida (`bigdata-net`) para hablar con los otros servicios.

### Servicio: Spark Master

```yaml
spark-master:
  image: apache/spark:3.5.1
  container_name: spark_master
  hostname: spark-master
  environment:
    - SPARK_MODE=master
    - SPARK_RPC_AUTHENTICATION_ENABLED=no
    - SPARK_RPC_ENCRYPTION_ENABLED=no
    - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
    - SPARK_SSL_ENABLED=no
  command: /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master
  ports:
    - "8080:8080" # Web UI
    - "7077:7077" # Master URL
  volumes:
    - ./jars:/opt/spark/jars # Compartir drivers
  networks:
    - bigdata-net
```

**Que hace:**
Es el nodo coordinador ("cerebro") del cluster Spark.

- **Rol**: Gestiona los recursos disponibles y asigna las tareas de procesamiento a los Workers. No procesa datos pesados él mismo, solo orquesta.
- **Puertos**:
  - `8080`: Expone la interfaz web (Spark UI) donde puedo ver el estado del cluster, los workers conectados y los jobs en ejecución.
  - `7077`: Es el puerto de comunicación interna que usan los Workers y los Clientes (como Jupyter) para enviar trabajos al Master.
- **Red**: Necesita la red compartida para que el Worker pueda decir "Hola, estoy aquí" enviando mensajes a `spark-master:7077`.

### Servicio: Spark Worker

```yaml
spark-worker:
  image: apache/spark:3.5.1
  container_name: spark_worker
  environment:
    - SPARK_MODE=worker
    - SPARK_WORKER_MEMORY=1G
    - SPARK_WORKER_CORES=1
    # ... (configuraciones de seguridad desactivadas)
  command: /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
  depends_on:
    - spark-master
  volumes:
    - ./jars:/opt/spark/jars # Compartir drivers
  networks:
    - bigdata-net
```

**Que hace:**
Es el "músculo" del cluster. Es quien realmente ejecuta los cálculos.

- **Conexión**: En el comando de inicio (`command`) le especificamos explícitamente `spark://spark-master:7077`. Así sabe a quién reportarse.
- **Recursos**: Le asigné 1 Core (`SPARK_WORKER_CORES=1`) y 1GB de RAM (`SPARK_WORKER_MEMORY=1G`). Si agrego más workers (escalado horizontal), tendría más capacidad de procesamiento paralelo total.
- **Drivers**: Monta el volumen `./jars` igual que el master y jupyter para asegurarse de tener las mismas librerías (como el driver de PostgreSQL) disponibles.

### Otros servicios: JupyterLab

```yaml
jupyter-lab:
  image: jupyter/pyspark-notebook:latest
  container_name: jupyter_lab
  ports:
    - "8888:8888" # JupyterLab
    - "4040:4040" # Spark UI (Driver)
  environment:
    - JUPYTER_ENABLE_LAB=yes
    - JUPYTER_TOKEN=bigdata2024 # Contraseña fija
    - SPARK_MASTER=spark://spark-master:7077
    - SPARK_DRIVER_HOST=jupyter_lab
  # ...
```

**Que hace:**
Actúa como mi "Cliente" o "Driver" interactivo. Es donde escribo el código Python (PySpark).

- Se conecta al Master (`SPARK_MASTER`) para enviar el código a ejecutar.
- Expone el puerto `4040` para ver el detalle de la ejecución de _mi_ aplicación específica.

### Acceso y Credenciales

Para acceder a JupyterLab:

- **URL**: `http://localhost:8888`
- **Password/Token**: `bigdata2024` (Configurado en `docker-compose.yml`)

### Comandos de Ejecución (Importante)

Dado que nuestro entorno está en Docker, debemos ejecutar los scripts **dentro** del contenedor, no desde nuestro Windows local.

**1. Instalar dependencias nuevas:**

```powershell
docker exec -it jupyter_lab pip install -r /home/jovyan/work/requirements.txt
```

**2. Ejecutar Pipeline ETL:**

```powershell
docker exec -it jupyter_lab spark-submit /home/jovyan/work/src/pipeline.py
```

---

## 2.2 Healthchecks

**¿Qué son?**
Un healthcheck es un comando que Docker ejecuta periódicamente dentro del contenedor para preguntar: "¿Estás realmente listo para trabajar?".

**¿Por qué los necesitamos?**
Docker por defecto solo sabe si el contenedor "arrancó" (el proceso existe), pero no si la _aplicación_ está lista.

- **Ejemplo**: PostgreSQL puede tardar 10 segundos en iniciar y aceptar conexiones.
- **Sin healthcheck**: Si Spark o mi script de ingesta intenta conectarse en el segundo 2, fallará con un error de conexión, aunque Docker diga que la base de datos está "running".
- **Con healthcheck**: Podríamos configurar los servicios dependientes para que esperen (`condition: service_healthy`) hasta que Postgres diga "estoy listo" (select 1), evitando errores de inicio por condiciones de carrera (race conditions).

---

## 2.3 Evidencia: Captura Spark UI

![SparkUI](capturas/localhost_8080.png)

**Que se ve en la captura (esperado):**

- **URL**: `spark://spark-master:7077`.
- **Workers**: Debería aparecer `1` en la lista de "Alive Workers".
- **Status**: El estado del Worker debe ser `ALIVE`.
- **Resources**: Debería mostrar `1 Cores` y `1024.0 MiB Memory` disponibles.

---

## 2.4 Prompts utilizados para la infraestructura

### Prompt 1: Creación inicial

**Herramienta:** Agentic AI (Google DeepMind)

**Tu prompt exacto:**

```text
(Reconstruido del contexto de la sesión inicial)
Necesito configurar un proyecto de Big Data usando Docker.
Quiero tener un pipeline que use:
1. PostgreSQL para guardar datos.
2. Un cluster de Spark (Master y Worker) para procesar los datos.
3. JupyterLab para ejecutar notebooks y conectarse a Spark.
Por favor crea la estructura de carpetas y el archivo docker-compose.yml necesario.
```

**Que te devolvio:**
Generó un archivo `docker-compose.yml` inicial con los 4 servicios solicitados, usando imágenes estándar (`postgres:13`, `bitnami/spark` o `apache/spark`, `jupyter/pyspark-notebook`) y definió una red `bigdata-net`.

**Que tuviste que cambiar y por que:**

- Tuvimos problemas iniciales con las imágenes de `bitnami/spark` por temas de permisos de usuario (Running as root vs non-root).
- **Cambio**: Migramos a usar las imágenes oficiales `apache/spark:3.5.1` que resultaron más estables para este entorno.
- **Ajuste**: Añadimos volúmenes explícitos para compartir drivers (`./jars`) entre todos los contenedores, algo que no estaba en la primera versión básica pero fue necesario para conectar Spark con Postgres.

### Prompt 2: Refinamiento de Jupyter y Red

**Herramienta:** Agentic AI

**Tu prompt exacto:**

```text
(Reconstruido)
El contenedor de Jupyter tiene problemas de permisos para instalar librerías y no ve al Spark Master.
Asegúrate de que Jupyter esté en la misma red y configura las variables de entorno para que se conecte al master en spark://spark-master:7077.
Además, necesito persistir los notebooks en una carpeta local.
```

**Que te devolvio y que cambiaste:**
Actualizó la configuración de `jupyter-lab` añadiendo `SPARK_DRIVER_HOST=jupyter_lab` (crucial para que los Workers puedan "devolver la llamada" al Driver) y mapeó correctamente los volúmenes `./notebooks` y `./data`.

---

## 2.5 Recursos web consultados

| Recurso               | URL                                                        | Que aprendiste de el                                             |
| --------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| Docker Hub (Postgres) | https://hub.docker.com/_/postgres                          | Variables de entorno necesarias (POSTGRES_USER, DB)              |
| Spark Documentation   | https://spark.apache.org/docs/latest/spark-standalone.html | Puertos requeridos (8080, 7077) y configuración de Master/Worker |
| Jupyter Docker Stacks | https://jupyter-docker-stacks.readthedocs.io/              | Cómo configurar PySpark dentro del contenedor de Jupyter         |

---

## 2.6 Evidencia Adicional

### Docker Desktop: Contenedores corriendo

Aquí vemos todos los servicios del `docker-compose.yml` activos y funcionando.

![Docker Desktop](capturas/docker_desktop.png)

### Jupyter Lab: Notebook de Análisis

Evidencia del entorno de trabajo conectado a Spark y ejecutando código.

![Jupyter Notebook](capturas/jupyter_notebook.png)

### Spark Job UI: Detalle de Ejecución

Evidencia de los "Jobs" (tareas) de Spark completados correctamente durante la ejecución del Random Forest.

![Spark Jobs UI](capturas/spark_jobs_ui.png)

---

## 2.7 Concepto: Docker y Contenedores

**¿Cómo se une Docker con los Contenedores?**

Esta gráfica explica la relación entre la Imagen (la receta), el Contenedor (la tarta) y el Docker Engine (el horno).

![Diagrama Docker](capturas/docker_concept_mermaid.png)

**Explicación:**

1.  **Docker Engine**: Es el software que instalaste. Actúa como el intermediario entre tu sistema operativo y los contenedores.
2.  **Imágenes**: Son plantillas estáticas (como una clase en POO o una receta de cocina). En nuestro caso, `apache/spark` es la imagen.
3.  **Contenedores**: Son la versión "viva" de la imagen. Cuando le das "Play", Docker Engine toma la imagen y crea un proceso aislado (el contenedor). Nota cómo usamos la **misma** imagen de Spark para crear **dos** contenedores distintos (Master y Worker), simplemente cambiándoles la configuración al arrancar.

```

```
