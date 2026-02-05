from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Iniciando análisis Spark + ML...")
    
    # Init Spark
    spark = SparkSession.builder \
        .appName("Analisis_Gran_Juego") \
        .getOrCreate()
        
    # Rutas
    DATA_PATH = "/home/jovyan/work/data/processed/qog_great_game.parquet"
    OUTPUT_DIR = "/home/jovyan/work/notebooks"
    
    # 1. Cargar Datos
    print(f"Cargando datos desde {DATA_PATH}...")
    try:
        df = spark.read.parquet(DATA_PATH)
        print(f"Registros encontrados: {df.count()}")
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return

    # 2. Matriz de Correlación
    print("Generando Matriz de Correlación...")
    features = ['gle_cgdpc', 'wdi_lifexp', 'p_polity2', 'vdem_corr', 'wdi_expmil']
    try:
        df_corr = df.select(features).toPandas().dropna()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Matriz de Correlación: Factores Políticos vs Económicos")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/grafico_correlacion.png")
        print(f"Gráfico guardado en {OUTPUT_DIR}/grafico_correlacion.png")
        plt.close()
    except Exception as e:
        print(f"Error generando correlación: {e}")

    # 3. Random Forest
    print("Entrenando Modelo Random Forest...")
    try:
        # Preprocesamiento
        df_ml = df.select(features).dropna()
        assembler = VectorAssembler(inputCols=['wdi_lifexp', 'p_polity2', 'vdem_corr', 'wdi_expmil'], outputCol="features")
        
        # Modelo
        rf = RandomForestRegressor(featuresCol="features", labelCol="gle_cgdpc", numTrees=100, seed=42)
        pipeline = Pipeline(stages=[assembler, rf])
        
        # Entrenar
        model = pipeline.fit(df_ml)
        print("Modelo entrenado.")
        
        # Feature Importance
        rf_model = model.stages[-1]
        importances = rf_model.featureImportances
        feature_cols = ['wdi_lifexp', 'p_polity2', 'vdem_corr', 'wdi_expmil']
        
        importances_list = []
        for i, col_name in enumerate(feature_cols):
            importances_list.append({"Feature": col_name, "Importance": importances[i]})
            
        df_imp = pd.DataFrame(importances_list).sort_values(by="Importance", ascending=False)
        
        # Graficar
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=df_imp, palette="viridis")
        plt.title("Random Forest: Importancia de Variables en el Desarrollo")
        plt.xlabel("Importancia Relativa")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/grafico_feature_importance.png")
        print(f"Gráfico guardado en {OUTPUT_DIR}/grafico_feature_importance.png")
        plt.close()
        
    except Exception as e:
        print(f"Error en ML: {e}")
        
    spark.stop()
    print("Análisis completado.")

if __name__ == "__main__":
    main()
