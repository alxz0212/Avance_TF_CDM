import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel import compare
import os
import sys

# Redirigir stdout para capturar en logs si es necesario
# sys.stdout = open('econometric_debug.log', 'w')

print("Iniciando análisis econométrico...")

# 1. Cargar datos
try:
    df = pd.read_parquet('data/processed/qog_great_game.parquet')
    print(f"Datos cargados. Shape: {df.shape}")
except Exception as e:
    print(f"Error cargando datos: {e}")
    # Intento de cargar desde Raw si falla parquet (solo para referencia, el parquet deberia existir)
    exit()

# 2. Preprocesamiento para Panel Data
# El índice debe ser (Entidad, Tiempo)
# Asegurar que cname y year existen
if 'cname' not in df.columns or 'year' not in df.columns:
    print("Columnas 'cname' o 'year' no encontradas.")
    print(df.columns)
    exit()

df = df.set_index(['cname', 'year'])

# Variables de interés
dependent_var = 'gle_cgdpc'
exog_vars = ['wdi_lifexp', 'p_polity2', 'vdem_corr', 'wdi_expmil']

# Verificar existencia de columnas
missing_cols = [col for col in exog_vars + [dependent_var] if col not in df.columns]
if missing_cols:
    print(f"Faltan columnas en el dataset: {missing_cols}")
    exit()

# Limpieza de nulos
df_clean = df[[dependent_var] + exog_vars].dropna()
print(f"Datos limpios para análisis. Shape: {df_clean.shape}")

# Verificar varianza (si es 0 en alguna entidad, Fixed Effects falla)
std_devs = df_clean.groupby(level=0)[exog_vars].std()
print("Desviación estándar intra-entidad (promedio):")
print(std_devs.mean())

# Si alguna variable tiene varianza 0 casi siempre, hay que sacarla
# Para este ejercicio, intentamos seguir.

# 3. Modelos
res_fe = None
res_re = None

try:
    print("Ejecutando Efectos Fijos...")
    mod_fe = PanelOLS(df_clean[dependent_var], df_clean[exog_vars], entity_effects=True)
    res_fe = mod_fe.fit()
    print("Efectos Fijos completado.")
except Exception as e:
    print(f"Fallo en Efectos Fijos: {e}")

try:
    print("Ejecutando Efectos Aleatorios...")
    mod_re = RandomEffects(df_clean[dependent_var], df_clean[exog_vars])
    res_re = mod_re.fit()
    print("Efectos Aleatorios completado.")
except Exception as e:
    print(f"Fallo en Efectos Aleatorios: {e}")

# 4. Resultados y Test de Hausman
output_dir = 'notebooks'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'hausman_results.txt')

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("=== ANÁLISIS ECONOMÉTRICO: EFECTOS FIJOS VS ALEATORIOS ===\n\n")
    
    if res_fe:
        f.write("1. Modelo de Efectos Fijos (Entity Effects):\n")
        f.write(str(res_fe.summary))
        f.write("\n\n")
    else:
        f.write("1. Modelo de Efectos Fijos: FALLÓ\n\n")
    
    if res_re:
        f.write("2. Modelo de Efectos Aleatorios:\n")
        f.write(str(res_re.summary))
        f.write("\n\n")
    else:
        f.write("2. Modelo de Efectos Aleatorios: FALLÓ\n\n")
    
    if res_fe and res_re:
        try:
            comparison = compare({'Fixed Effects': res_fe, 'Random Effects': res_re})
            f.write("3. Comparación de Modelos:\n")
            f.write(str(comparison.summary))
            f.write("\n\n")
            
            f.write("=== INTERPRETACIÓN (TEST DE HAUSMAN) ===\n")
            f.write("El Test de Hausman compara si los coeficientes de ambos modelos divergen sistemáticamente.\n")
            f.write("Si el p-value es bajo (< 0.05), se prefiere Efectos Fijos (Fixed Effects) porque los efectos únicos de cada país están correlacionados con las variables.\n")
            f.write("Si el p-value es alto (> 0.05), se prefiere Efectos Aleatorios (Random Effects) por ser más eficiente.\n")
        except Exception as e:
            f.write(f"No se pudo realizar la comparación automática: {e}\n")
    else:
        f.write("No se pudo realizar la comparación porque uno de los modelos falló.\n")

print(f"Análisis completado. Resultados guardados en {output_path}")
