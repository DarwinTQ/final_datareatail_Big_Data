"""
04_mapreduce.py  -  Modelo MapReduce: productos más vendidos en un período (DataRetail)

Se ejecuta dentro del contenedor de Spark:
    docker exec -it spark-jupyter spark-submit --driver-memory 8g work/04_mapreduce.py
    (opcional) ... work/04_mapreduce.py --inicio 2020-06-01 --fin 2020-08-31

PARTE 1  MapReduce "clásico" paso a paso (Python puro, como Hadoop):
         INPUT SPLITS -> MAP (+ combiner) -> SHUFFLE & SORT -> REDUCE
PARTE 2  El mismo job en Spark (RDD map / reduceByKey) leyendo desde HDFS
PARTE 3  Tendencia mensual por categoría con MapReduce (clave compuesta mes+categoría)
"""
import argparse
import csv
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from operator import add

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_LOCAL = os.path.join(BASE, "data", "processed", "ventas.csv")
HDFS_VENTAS = "hdfs://namenode:9000/dataretail/raw/ventas.csv"
HDFS_SALIDA = "hdfs://namenode:9000/dataretail/curated/mapreduce_top_productos"
DIR_OUT = os.path.join(BASE, "resultados", "mapreduce")
os.makedirs(DIR_OUT, exist_ok=True)

# Columnas de ventas.csv
I_FECHA, I_PRODUCTO, I_CATEGORIA, I_CANTIDAD = 1, 3, 4, 7
N_SPLITS = 4          # como si el archivo estuviera en 4 bloques de HDFS
TOP_N = 10


def titulo(t):
    print("\n" + "=" * 72 + f"\n {t}\n" + "=" * 72)


# ===========================================================================
# PARTE 1: MapReduce clásico en Python puro
# ===========================================================================
def calcular_splits(ruta, n):
    """Divide el archivo en n rangos de bytes, igual que Hadoop crea InputSplits."""
    tam = os.path.getsize(ruta)
    paso = tam // n
    return [(i * paso, tam if i == n - 1 else (i + 1) * paso) for i in range(n)]


def leer_split(ruta, inicio, fin):
    """Lee las líneas de un split. Si empieza a mitad de una línea, la salta
    (esa línea la procesa el split anterior), tal como hace Hadoop."""
    with open(ruta, "rb") as f:
        f.seek(inicio)
        if inicio == 0:
            f.readline()                     # salta la cabecera del CSV
        else:
            f.readline()                     # salta la línea incompleta
        while f.tell() <= fin:
            linea = f.readline()
            if not linea:
                break
            yield linea.decode("utf-8")


def mapper(linea, inicio_periodo, fin_periodo):
    """MAP: por cada venta del período emite (producto, cantidad)."""
    campos = next(csv.reader([linea]))
    fecha = campos[I_FECHA][:10]             # 'YYYY-MM-DD'
    if inicio_periodo <= fecha <= fin_periodo:
        yield campos[I_PRODUCTO], int(campos[I_CANTIDAD])


def tarea_map(args):
    """Una tarea MAP por split. Incluye un COMBINER (suma local) que reduce
    la cantidad de datos enviados a la fase shuffle."""
    ruta, (ini, fin), p_ini, p_fin, num = args
    t0 = time.perf_counter()
    lineas, emitidos = 0, 0
    combinado = Counter()
    for linea in leer_split(ruta, ini, fin):
        lineas += 1
        for clave, valor in mapper(linea, p_ini, p_fin):
            emitidos += 1
            combinado[clave] += valor        # combiner
    return {"split": num, "bytes": f"{ini:,} - {fin:,}", "lineas_leidas": lineas,
            "pares_emitidos": emitidos, "pares_tras_combiner": len(combinado),
            "segundos": round(time.perf_counter() - t0, 2), "salida": combinado}


def shuffle_sort(salidas_map):
    """SHUFFLE & SORT: agrupa todos los valores de una misma clave y ordena por clave."""
    grupos = defaultdict(list)
    for salida in salidas_map:
        for clave, valor in salida.items():
            grupos[clave].append(valor)
    return dict(sorted(grupos.items()))


def reducer(clave, valores):
    """REDUCE: suma las cantidades de cada producto."""
    return clave, sum(valores)


def mapreduce_python(p_ini, p_fin):
    titulo(f"PARTE 1: MAPREDUCE CLÁSICO (Python puro)  |  Período {p_ini} a {p_fin}")
    t0 = time.perf_counter()

    splits = calcular_splits(ARCHIVO_LOCAL, N_SPLITS)
    print(f"\n[1] INPUT SPLITS: archivo de {os.path.getsize(ARCHIVO_LOCAL) / 1e6:,.1f} MB "
          f"dividido en {N_SPLITS} splits")
    for i, (a, b) in enumerate(splits):
        print(f"    split {i}: bytes {a:,} -> {b:,}")

    print(f"\n[2] MAP + COMBINER: {N_SPLITS} mappers en paralelo")
    with Pool(N_SPLITS) as pool:
        resultados = pool.map(tarea_map, [(ARCHIVO_LOCAL, s, p_ini, p_fin, i)
                                          for i, s in enumerate(splits)])
    tabla = pd.DataFrame([{k: v for k, v in r.items() if k != "salida"} for r in resultados])
    print(tabla.to_string(index=False))
    tabla.to_csv(os.path.join(DIR_OUT, "p1_fase_map.csv"), index=False)
    total_emit = tabla["pares_emitidos"].sum()
    total_comb = tabla["pares_tras_combiner"].sum()
    print(f"    Pares emitidos por los mappers: {total_emit:,}  ->  tras combiner: {total_comb:,} "
          f"({(1 - total_comb / max(total_emit, 1)) * 100:.1f}% menos tráfico de red)")
    con_datos = [r for r in resultados if r["salida"]]
    if con_datos:
        r = con_datos[0]
        print(f"    Ejemplo de salida del mapper {r['split']}: {list(r['salida'].items())[:3]}")

    print("\n[3] SHUFFLE & SORT: agrupando valores por clave (producto)")
    grupos = shuffle_sort([r["salida"] for r in resultados])
    print(f"    Claves únicas: {len(grupos):,}")
    for clave, valores in list(grupos.items())[:3]:
        print(f"    {clave!r:35} -> {valores}")

    print("\n[4] REDUCE: sumando cantidades por producto")
    reducido = [reducer(k, v) for k, v in grupos.items()]
    top = sorted(reducido, key=lambda x: -x[1])[:TOP_N]
    seg = time.perf_counter() - t0
    df_top = pd.DataFrame(top, columns=["producto", "unidades_vendidas"])
    df_top.index = range(1, len(df_top) + 1)
    print(f"\n    TOP {TOP_N} PRODUCTOS MÁS VENDIDOS ({p_ini} a {p_fin}):")
    print(df_top.to_string())
    print(f"\n    Tiempo total MapReduce Python: {seg:.2f} s")
    df_top.to_csv(os.path.join(DIR_OUT, "p1_top_productos_python.csv"), index_label="puesto")
    return df_top, seg


# ===========================================================================
# PARTE 2: el mismo job con Spark (MapReduce distribuido sobre HDFS)
# ===========================================================================
def fuente_spark(sc):
    try:
        jvm, conf = sc._jvm, sc._jsc.hadoopConfiguration()
        path = jvm.org.apache.hadoop.fs.Path(HDFS_VENTAS)
        if path.getFileSystem(conf).exists(path):
            return HDFS_VENTAS, True
    except Exception:
        pass
    return "file://" + ARCHIVO_LOCAL, False


def borrar_hdfs(sc, ruta):
    jvm, conf = sc._jvm, sc._jsc.hadoopConfiguration()
    path = jvm.org.apache.hadoop.fs.Path(ruta)
    fs = path.getFileSystem(conf)
    if fs.exists(path):
        fs.delete(path, True)


def parsear(linea):
    return next(csv.reader([linea]))


def mapreduce_spark(spark, p_ini, p_fin):
    titulo(f"PARTE 2: MAPREDUCE CON SPARK (RDD)  |  Período {p_ini} a {p_fin}")
    sc = spark.sparkContext
    ruta, en_hdfs = fuente_spark(sc)
    print(f"  Fuente: {ruta}")
    t0 = time.perf_counter()

    lineas = sc.textFile(ruta)
    cabecera = lineas.first()
    print(f"  Particiones de entrada (≈ bloques HDFS): {lineas.getNumPartitions()}")

    conteo = (lineas.filter(lambda l: l != cabecera)
              .map(parsear)                                                    # parseo
              .filter(lambda c: p_ini <= c[I_FECHA][:10] <= p_fin)             # período
              .map(lambda c: (c[I_PRODUCTO], int(c[I_CANTIDAD])))               # MAP
              .reduceByKey(add))                                                # SHUFFLE + REDUCE
    top = conteo.takeOrdered(TOP_N, key=lambda x: -x[1])
    seg = time.perf_counter() - t0

    df_top = pd.DataFrame(top, columns=["producto", "unidades_vendidas"])
    df_top.index = range(1, len(df_top) + 1)
    print(df_top.to_string())
    print(f"\n  Tiempo total MapReduce Spark: {seg:.2f} s")
    print("\n  Linaje del RDD (plan de ejecución: map -> shuffle -> reduce):")
    print("  " + conteo.toDebugString().decode("utf-8").replace("\n", "\n  "))
    df_top.to_csv(os.path.join(DIR_OUT, "p2_top_productos_spark.csv"), index_label="puesto")

    if en_hdfs:
        try:
            borrar_hdfs(sc, HDFS_SALIDA)
            (conteo.sortBy(lambda x: -x[1]).map(lambda x: f"{x[0]}\t{x[1]}")
             .coalesce(1).saveAsTextFile(HDFS_SALIDA))
            print(f"\n  Resultado completo guardado en HDFS: {HDFS_SALIDA}/part-00000")
        except Exception as e:
            print("  No se pudo guardar en HDFS:", str(e)[:150])
    return df_top, seg, ruta


# ===========================================================================
# PARTE 3: tendencia mensual (clave compuesta)
# ===========================================================================
def tendencia_mensual(spark, ruta):
    titulo("PARTE 3: TENDENCIA MENSUAL POR CATEGORÍA (MapReduce con clave compuesta)")
    sc = spark.sparkContext
    lineas = sc.textFile(ruta)
    cabecera = lineas.first()
    pares = (lineas.filter(lambda l: l != cabecera).map(parsear)
             .map(lambda c: ((c[I_FECHA][:7], c[I_CATEGORIA]), int(c[I_CANTIDAD])))  # MAP
             .reduceByKey(add)                                                        # REDUCE
             .collect())
    df = pd.DataFrame([(m, c, u) for (m, c), u in pares], columns=["mes", "categoria", "unidades"])
    top_cat = df.groupby("categoria")["unidades"].sum().nlargest(5).index
    pivote = (df[df["categoria"].isin(top_cat)]
              .pivot(index="mes", columns="categoria", values="unidades").fillna(0).sort_index())
    print(pivote.astype(int).to_string())
    pivote.to_csv(os.path.join(DIR_OUT, "p3_tendencia_mensual_categorias.csv"))

    # Crecimiento: último mes completo vs. primer mes
    if len(pivote) >= 2:
        crec = ((pivote.iloc[-2] - pivote.iloc[0]) / pivote.iloc[0].replace(0, 1) * 100).round(1)
        print("\n  Variación % (primer mes vs. penúltimo mes):")
        print(crec.sort_values(ascending=False).to_string())

    ax = pivote.plot(figsize=(11, 5.5), marker="o")
    ax.set_title("Tendencia mensual de unidades vendidas - Top 5 categorías (MapReduce)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Mes"); ax.set_ylabel("Unidades"); ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(DIR_OUT, "p3_tendencia_mensual.png"), dpi=150); plt.close()
    print("\n  Gráfico: resultados/mapreduce/p3_tendencia_mensual.png")


def grafico_top(df_top, p_ini, p_fin):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(df_top["producto"][::-1], df_top["unidades_vendidas"][::-1], color="#1f5fa8")
    ax.set_title(f"Top {TOP_N} productos más vendidos ({p_ini} a {p_fin}) - MapReduce",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Unidades"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_OUT, "p2_top_productos.png"), dpi=150); plt.close(fig)


# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", default="2020-06-01", help="YYYY-MM-DD")
    ap.add_argument("--fin", default="2020-08-31", help="YYYY-MM-DD")
    args = ap.parse_args()

    top_py, t_py = mapreduce_python(args.inicio, args.fin)
    if top_py.empty:
        print("\n  ⚠ No hay ventas en ese período. Prueba otras fechas con --inicio y --fin.")

    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName("DataRetail - MapReduce")
             .master("local[8]").config("spark.sql.shuffle.partitions", "16").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    top_sp, t_sp, ruta = mapreduce_spark(spark, args.inicio, args.fin)
    if not top_sp.empty:
        grafico_top(top_sp, args.inicio, args.fin)
    tendencia_mensual(spark, ruta)

    titulo("COMPARACIÓN")
    iguales = top_py.equals(top_sp)
    print(f"  Resultados idénticos en ambos métodos: {'SÍ' if iguales else 'NO'}")
    print(f"  MapReduce Python puro ({N_SPLITS} procesos): {t_py:.2f} s")
    print(f"  MapReduce en Spark (local[8]):          {t_sp:.2f} s")
    pd.DataFrame({"metodo": ["Python puro (MapReduce clásico)", "Apache Spark (RDD)"],
                  "segundos": [round(t_py, 2), round(t_sp, 2)]}).to_csv(
        os.path.join(DIR_OUT, "comparacion_tiempos.csv"), index=False)
    print("\nListo. Resultados en resultados/mapreduce/")
    spark.stop()
