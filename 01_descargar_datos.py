"""
01_descargar_datos.py
Descarga el dataset real de Kaggle (compras en una tienda de electrónicos)
y lo copia a la carpeta del proyecto: data/raw/
Requisitos: pip install kagglehub pandas
"""
import glob
import os
import shutil

import kagglehub
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DESTINO = os.path.join(BASE, "data", "raw")
os.makedirs(DESTINO, exist_ok=True)

# 1. Descargar (kagglehub lo guarda en su caché)
path = kagglehub.dataset_download("mkechinov/ecommerce-purchase-history-from-electronics-store")
print("Descargado en caché:", path)

# 2. Copiar a la carpeta del proyecto
for archivo in glob.glob(os.path.join(path, "*")):
    shutil.copy(archivo, DESTINO)
    print("Copiado a data/raw:", os.path.basename(archivo))

# 3. Exploración rápida (tómale captura para el Word)
csv = sorted(glob.glob(os.path.join(DESTINO, "*.csv")))[0]
df = pd.read_csv(csv, dtype={"order_id": str, "product_id": str,
                             "category_id": str, "user_id": str})
print("\n=== Dimensiones (filas, columnas) ===")
print(df.shape)
print("\n=== Tipos de datos ===")
print(df.dtypes)
print("\n=== Primeras filas ===")
print(df.head(10).to_string())
print("\n=== Valores nulos por columna ===")
print(df.isnull().sum())
print("\n=== Estadísticas de precio ===")
print(df["price"].describe())
