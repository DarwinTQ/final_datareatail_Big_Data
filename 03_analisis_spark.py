"""
03_analisis_spark.py  -  Análisis de patrones de ventas con Apache Spark (DataRetail)

Se ejecuta DENTRO del contenedor de Spark:
    docker exec -it spark-jupyter spark-submit work/03_analisis_spark.py

Lee los datos desde HDFS (hdfs://namenode:9000/dataretail/raw). Si HDFS no está
disponible, usa la copia local data/processed. Genera:
    resultados/tablas/*.csv     tablas de resultados para el Word
    resultados/graficos/*.png   gráficos para el Word
    hdfs:///dataretail/curated  resultados en formato Parquet (capa "curated")
"""
import os

import matplotlib
matplotlib.use("Agg")                      # sin pantalla: guarda imágenes a archivo
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (BooleanType, DoubleType, IntegerType, StringType,
                               StructField, StructType, TimestampType)

BASE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(BASE, "data", "processed")
HDFS = "hdfs://namenode:9000/dataretail"
DIR_TABLAS = os.path.join(BASE, "resultados", "tablas")
DIR_GRAF = os.path.join(BASE, "resultados", "graficos")
os.makedirs(DIR_TABLAS, exist_ok=True)
os.makedirs(DIR_GRAF, exist_ok=True)

spark = (SparkSession.builder
         .appName("DataRetail - Analisis de ventas")
         .master("local[8]")                              # 8 núcleos en paralelo
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.sql.shuffle.partitions", "16")    # 200 por defecto es excesivo en 1 nodo
         .config("spark.sql.adaptive.enabled", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Esquemas explícitos (más rápido y seguro que inferSchema)
# ---------------------------------------------------------------------------
ESQ_VENTAS = StructType([
    StructField("id_transaccion", StringType()), StructField("fecha", TimestampType()),
    StructField("id_producto", StringType()), StructField("producto", StringType()),
    StructField("categoria", StringType()), StructField("marca", StringType()),
    StructField("precio", DoubleType()), StructField("cantidad", IntegerType()),
    StructField("tienda", StringType()), StructField("canal", StringType()),
    StructField("metodo_pago", StringType()), StructField("id_cliente", StringType())])
ESQ_CLICKS = StructType([
    StructField("id_sesion", StringType()), StructField("id_usuario", StringType()),
    StructField("fecha_hora", TimestampType()), StructField("producto_visitado", StringType()),
    StructField("id_producto", StringType()), StructField("categoria", StringType()),
    StructField("tiempo_navegacion_seg", DoubleType()), StructField("paginas_vistas", IntegerType()),
    StructField("dispositivo", StringType()), StructField("origen_trafico", StringType()),
    StructField("agrego_carrito", BooleanType()), StructField("compro", BooleanType())])
ESQ_IOT = StructType([
    StructField("tienda", StringType()), StructField("fecha_hora", TimestampType()),
    StructField("sensor_id", StringType()), StructField("clientes_detectados", IntegerType()),
    StructField("temperatura_c", DoubleType()), StructField("humedad_pct", DoubleType()),
    StructField("alerta_temperatura", BooleanType())])


def ruta(nombre):
    """Usa HDFS si el archivo existe allí; si no, la copia local."""
    ruta_hdfs = f"{HDFS}/raw/{nombre}"
    try:
        jvm = spark.sparkContext._jvm
        conf = spark.sparkContext._jsc.hadoopConfiguration()
        path = jvm.org.apache.hadoop.fs.Path(ruta_hdfs)
        if path.getFileSystem(conf).exists(path):
            print(f"  Leyendo desde HDFS: {ruta_hdfs}")
            return ruta_hdfs
    except Exception:
        pass
    print(f"  HDFS no disponible, leyendo local: {nombre}")
    return "file://" + os.path.join(LOCAL, nombre)


def leer_csv(nombre, esquema):
    return spark.read.csv(ruta(nombre), header=True, schema=esquema)


def titulo(texto):
    print("\n" + "=" * 70 + f"\n {texto}\n" + "=" * 70)


def guardar(df_spark, nombre, mostrar=15):
    """Muestra el resultado, lo guarda como CSV local y devuelve un pandas DataFrame."""
    df_spark.show(mostrar, truncate=False)
    pdf = df_spark.toPandas()
    pdf.to_csv(os.path.join(DIR_TABLAS, f"{nombre}.csv"), index=False, encoding="utf-8")
    return pdf


def grafico(pdf, x, y, nombre, titulo_g, tipo="bar", horizontal=False, xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if tipo == "line":
        ax.plot(pdf[x].astype(str), pdf[y], marker="o", color="#1f5fa8")
        plt.xticks(rotation=45, ha="right")
    elif horizontal:
        ax.barh(pdf[x].astype(str)[::-1], pdf[y][::-1], color="#1f5fa8")
    else:
        ax.bar(pdf[x].astype(str), pdf[y], color="#1f5fa8")
        plt.xticks(rotation=45, ha="right")
    ax.set_title(titulo_g, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x" if horizontal else "y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_GRAF, f"{nombre}.png"), dpi=150)
    plt.close(fig)
    print(f"  Gráfico guardado: resultados/graficos/{nombre}.png")


# ---------------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------------
titulo("CARGA DE DATOS")
ventas = leer_csv("ventas.csv", ESQ_VENTAS).withColumn("importe", F.col("precio") * F.col("cantidad"))
clicks = leer_csv("ecommerce_clicks.csv", ESQ_CLICKS)
iot = leer_csv("iot_sensores.csv", ESQ_IOT)
redes = (spark.read.option("multiLine", True).json(ruta("redes_sociales.json"))
         .withColumn("fecha", F.to_timestamp(F.col("fecha").getField("$date"))))
ventas.cache()
print(f"  Ventas: {ventas.count():,} | Clics: {clicks.count():,} | "
      f"IoT: {iot.count():,} | Comentarios: {redes.count():,}")
ventas.printSchema()

# ---------------------------------------------------------------------------
# 1. RESUMEN GENERAL (KPIs)
# ---------------------------------------------------------------------------
titulo("1. KPIs GENERALES")
guardar(ventas.agg(
    F.countDistinct("id_transaccion").alias("transacciones"),
    F.sum("cantidad").alias("unidades_vendidas"),
    F.round(F.sum("importe"), 2).alias("ingresos_totales"),
    F.round(F.sum("importe") / F.countDistinct("id_transaccion"), 2).alias("ticket_promedio"),
    F.countDistinct("id_cliente").alias("clientes_unicos"),
    F.countDistinct("id_producto").alias("productos_distintos"),
    F.min("fecha").alias("desde"), F.max("fecha").alias("hasta")), "01_kpis")

# ---------------------------------------------------------------------------
# 2. TOP 10 PRODUCTOS Y CATEGORÍAS
# ---------------------------------------------------------------------------
titulo("2. TOP 10 PRODUCTOS MÁS VENDIDOS")
top_prod = guardar(ventas.groupBy("producto")
                   .agg(F.sum("cantidad").alias("unidades"), F.round(F.sum("importe"), 2).alias("ingresos"))
                   .orderBy(F.desc("unidades")).limit(10), "02_top_productos")
grafico(top_prod, "producto", "unidades", "02_top_productos",
        "Top 10 productos más vendidos", horizontal=True, xlabel="Unidades")

titulo("3. INGRESOS POR CATEGORÍA")
cat = guardar(ventas.groupBy("categoria")
              .agg(F.round(F.sum("importe"), 2).alias("ingresos"), F.sum("cantidad").alias("unidades"))
              .orderBy(F.desc("ingresos")).limit(10), "03_categorias")
grafico(cat, "categoria", "ingresos", "03_categorias",
        "Top 10 categorías por ingresos", horizontal=True, xlabel="Ingresos")

titulo("4. TOP 10 MARCAS")
guardar(ventas.groupBy("marca").agg(F.round(F.sum("importe"), 2).alias("ingresos"),
                                    F.sum("cantidad").alias("unidades"))
        .orderBy(F.desc("ingresos")).limit(10), "04_marcas")

# ---------------------------------------------------------------------------
# 3. TENDENCIAS EN EL TIEMPO
# ---------------------------------------------------------------------------
titulo("5. EVOLUCIÓN MENSUAL DE VENTAS")
mes = guardar(ventas.withColumn("mes", F.date_format("fecha", "yyyy-MM"))
              .groupBy("mes").agg(F.round(F.sum("importe"), 2).alias("ingresos"),
                                  F.countDistinct("id_transaccion").alias("transacciones"))
              .orderBy("mes"), "05_ventas_mensuales", mostrar=24)
grafico(mes, "mes", "ingresos", "05_ventas_mensuales", "Evolución mensual de ingresos",
        tipo="line", ylabel="Ingresos")

titulo("6. VENTAS POR HORA DEL DÍA (horas pico)")
hora = guardar(ventas.withColumn("hora", F.hour("fecha")).groupBy("hora")
               .agg(F.countDistinct("id_transaccion").alias("transacciones")).orderBy("hora"),
               "06_ventas_por_hora", mostrar=24)
grafico(hora, "hora", "transacciones", "06_ventas_por_hora", "Transacciones por hora del día",
        xlabel="Hora", ylabel="Transacciones")

titulo("7. VENTAS POR DÍA DE LA SEMANA")
dia = guardar(ventas.withColumn("num_dia", F.dayofweek("fecha"))
              .withColumn("dia", F.date_format("fecha", "EEEE"))
              .groupBy("num_dia", "dia").agg(F.round(F.sum("importe"), 2).alias("ingresos"))
              .orderBy("num_dia"), "07_ventas_por_dia")
grafico(dia, "dia", "ingresos", "07_ventas_por_dia", "Ingresos por día de la semana", ylabel="Ingresos")

# ---------------------------------------------------------------------------
# 4. CANALES, TIENDAS Y MÉTODOS DE PAGO
# ---------------------------------------------------------------------------
titulo("8. INGRESOS POR TIENDA / CANAL")
tienda = guardar(ventas.groupBy("canal", "tienda")
                 .agg(F.round(F.sum("importe"), 2).alias("ingresos"),
                      F.countDistinct("id_transaccion").alias("transacciones"),
                      F.round(F.sum("importe") / F.countDistinct("id_transaccion"), 2).alias("ticket_promedio"))
                 .orderBy(F.desc("ingresos")), "08_tiendas")
grafico(tienda, "tienda", "ingresos", "08_tiendas", "Ingresos por tienda y e-commerce", ylabel="Ingresos")

titulo("9. MÉTODOS DE PAGO POR CANAL")
pago = guardar(ventas.groupBy("canal", "metodo_pago")
               .agg(F.countDistinct("id_transaccion").alias("transacciones"))
               .orderBy("canal", F.desc("transacciones")), "09_metodos_pago")
pv = pago.pivot(index="metodo_pago", columns="canal", values="transacciones").fillna(0)
ax = pv.plot(kind="bar", figsize=(10, 5.5), color=["#1f5fa8", "#f0a030"])
ax.set_title("Métodos de pago por canal", fontsize=13, fontweight="bold")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(DIR_GRAF, "09_metodos_pago.png"), dpi=150); plt.close()

# ---------------------------------------------------------------------------
# 5. CLIENTES: segmentación RFM simple (Recencia, Frecuencia, Monto)
# ---------------------------------------------------------------------------
titulo("10. SEGMENTACIÓN DE CLIENTES (RFM)")
fecha_ref = ventas.agg(F.max("fecha")).first()[0]
rfm = (ventas.groupBy("id_cliente")
       .agg(F.datediff(F.lit(fecha_ref), F.max("fecha")).alias("recencia_dias"),
            F.countDistinct("id_transaccion").alias("frecuencia"),
            F.round(F.sum("importe"), 2).alias("monto"))
       .withColumn("segmento",
                   F.when((F.col("frecuencia") >= 3) & (F.col("recencia_dias") <= 60), "VIP / Leal")
                    .when(F.col("frecuencia") >= 2, "Recurrente")
                    .when(F.col("recencia_dias") <= 60, "Nuevo")
                    .otherwise("En riesgo / Inactivo")))
seg = guardar(rfm.groupBy("segmento")
              .agg(F.count("*").alias("clientes"), F.round(F.avg("monto"), 2).alias("gasto_promedio"),
                   F.round(F.sum("monto"), 2).alias("ingresos"))
              .orderBy(F.desc("ingresos")), "10_segmentos_rfm")
grafico(seg, "segmento", "clientes", "10_segmentos_rfm", "Clientes por segmento RFM", ylabel="Clientes")

# ---------------------------------------------------------------------------
# 6. PRODUCTOS QUE SE COMPRAN JUNTOS (canasta)
# ---------------------------------------------------------------------------
titulo("11. PARES DE CATEGORÍAS COMPRADAS EN LA MISMA ORDEN")
a = ventas.select("id_transaccion", F.col("categoria").alias("cat_a")).distinct()
b = ventas.select("id_transaccion", F.col("categoria").alias("cat_b")).distinct()
guardar(a.join(b, "id_transaccion").where(F.col("cat_a") < F.col("cat_b"))
        .groupBy("cat_a", "cat_b").agg(F.count("*").alias("veces_juntos"))
        .orderBy(F.desc("veces_juntos")).limit(10), "11_canasta")

# ---------------------------------------------------------------------------
# 7. E-COMMERCE: embudo de conversión
# ---------------------------------------------------------------------------
titulo("12. EMBUDO E-COMMERCE POR DISPOSITIVO")
guardar(clicks.groupBy("dispositivo")
        .agg(F.count("*").alias("sesiones"),
             F.round(F.avg("tiempo_navegacion_seg"), 1).alias("tiempo_prom_seg"),
             F.sum(F.col("agrego_carrito").cast("int")).alias("agregaron_carrito"),
             F.sum(F.col("compro").cast("int")).alias("compras"))
        .withColumn("tasa_conversion_%", F.round(F.col("compras") / F.col("sesiones") * 100, 2))
        .withColumn("abandono_carrito_%",
                    F.round((1 - F.col("compras") / F.col("agregaron_carrito")) * 100, 2))
        .orderBy(F.desc("sesiones")), "12_embudo_dispositivo")

titulo("13. CONVERSIÓN POR ORIGEN DE TRÁFICO")
orig = guardar(clicks.groupBy("origen_trafico")
               .agg(F.count("*").alias("sesiones"), F.sum(F.col("compro").cast("int")).alias("compras"))
               .withColumn("tasa_conversion_%", F.round(F.col("compras") / F.col("sesiones") * 100, 2))
               .orderBy(F.desc("tasa_conversion_%")), "13_origen_trafico")
grafico(orig, "origen_trafico", "tasa_conversion_%", "13_origen_trafico",
        "Tasa de conversión por origen de tráfico (%)", ylabel="%")

# ---------------------------------------------------------------------------
# 8. REDES SOCIALES: sentimiento y palabras clave
# ---------------------------------------------------------------------------
titulo("14. SENTIMIENTO POR RED SOCIAL")
guardar(redes.groupBy("red_social").pivot("sentimiento", ["positivo", "neutral", "negativo"])
        .count().fillna(0).orderBy("red_social"), "14_sentimiento_redes")

titulo("15. PALABRAS CLAVE MÁS FRECUENTES EN COMENTARIOS NEGATIVOS")
neg = guardar(redes.where(F.col("sentimiento") == "negativo")
              .select(F.explode(F.expr("slice(palabras_clave, 1, size(palabras_clave) - 2)")).alias("palabra"))  # sin marca/categoría
              .groupBy("palabra").count().orderBy(F.desc("count")).limit(10), "15_palabras_negativas")
grafico(neg, "palabra", "count", "15_palabras_negativas",
        "Temas de quejas más frecuentes", horizontal=True, xlabel="Menciones")

# ---------------------------------------------------------------------------
# 9. IoT + VENTAS: afluencia vs. transacciones en tiendas físicas
# ---------------------------------------------------------------------------
titulo("16. AFLUENCIA (IoT) VS. VENTAS POR TIENDA")
ventas_hora = (ventas.where(F.col("canal") == "Tienda física")
               .withColumn("fecha_hora", F.date_trunc("hour", "fecha"))
               .groupBy("tienda", "fecha_hora")
               .agg(F.countDistinct("id_transaccion").alias("transacciones")))
cruce = (iot.join(ventas_hora, ["tienda", "fecha_hora"], "left").fillna(0, ["transacciones"]))
guardar(cruce.groupBy("tienda")
        .agg(F.sum("clientes_detectados").alias("visitantes"),
             F.sum("transacciones").alias("transacciones"),
             F.round(F.avg("temperatura_c"), 1).alias("temp_promedio"),
             F.sum(F.col("alerta_temperatura").cast("int")).alias("alertas_temperatura"))
        .withColumn("conversion_%", F.round(F.col("transacciones") / F.col("visitantes") * 100, 2))
        .orderBy(F.desc("visitantes")), "16_iot_vs_ventas")

titulo("17. AFLUENCIA PROMEDIO POR HORA (sensores IoT)")
afl = guardar(iot.withColumn("hora", F.hour("fecha_hora")).groupBy("hora")
              .agg(F.round(F.avg("clientes_detectados"), 1).alias("clientes_promedio"))
              .orderBy("hora"), "17_afluencia_hora", mostrar=24)
grafico(afl, "hora", "clientes_promedio", "17_afluencia_hora",
        "Clientes detectados por hora (promedio)", xlabel="Hora", ylabel="Clientes")

# ---------------------------------------------------------------------------
# 10. Guardar capa "curated" en HDFS (formato Parquet, columnar y comprimido)
# ---------------------------------------------------------------------------
titulo("18. GUARDANDO RESULTADOS EN HDFS (Parquet)")
try:
    (ventas.withColumn("anio_mes", F.date_format("fecha", "yyyy-MM"))
     .write.mode("overwrite").partitionBy("anio_mes").parquet(f"{HDFS}/curated/ventas"))
    rfm.write.mode("overwrite").parquet(f"{HDFS}/curated/clientes_rfm")
    print(f"  OK -> {HDFS}/curated/ventas (particionado por mes)")
    print(f"  OK -> {HDFS}/curated/clientes_rfm")
except Exception as e:
    print("  No se pudo escribir en HDFS (¿está levantado el clúster?):", str(e)[:200])

print("\nAnálisis terminado. Revisa la carpeta 'resultados/' en tu proyecto.")
spark.stop()