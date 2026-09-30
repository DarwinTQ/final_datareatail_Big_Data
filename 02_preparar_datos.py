"""
02_preparar_datos.py
Limpia el dataset real de Kaggle y genera las demás fuentes de DataRetail:

  data/processed/
    ventas.csv            real + enriquecido (tienda, canal, método de pago)
    ecommerce_clicks.csv  simulado: navegación en la web
    redes_sociales.json   simulado: comentarios de clientes
    iot_sensores.csv      simulado: sensores en tiendas físicas
    clientes.json         para MongoDB (con resumen de compras y consentimiento)
    ventas_muestra.json   para MongoDB (documentos anidados)

Requisitos: pip install pandas numpy faker
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
from faker import Faker

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data", "processed")
os.makedirs(OUT, exist_ok=True)

SEMILLA = 42
rng = np.random.default_rng(SEMILLA)
Faker.seed(SEMILLA)
fake = Faker("es_MX")

TIENDAS = ["Tienda Miraflores", "Tienda San Isidro", "Tienda Jockey Plaza",
           "Tienda Mega Plaza", "Tienda Arequipa"]
DISTRITOS = ["Miraflores", "San Isidro", "Surco", "La Molina", "San Borja",
             "Jesús María", "Lince", "Los Olivos", "San Miguel", "Chorrillos",
             "Ate", "Comas", "Callao", "Pueblo Libre", "Magdalena"]
MAX_CLIENTES_MONGO = 50_000
N_CLICKS = 300_000
N_COMENTARIOS = 8_000
N_VENTAS_MONGO = 20_000


def fecha_mongo(ts):
    """Formato Extended JSON: MongoDB lo importa como tipo Date."""
    return {"$date": pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")}


def fechas_aleatorias(inicio, fin, n):
    segundos = int((fin - inicio).total_seconds())
    return inicio + pd.to_timedelta(rng.integers(0, segundos, n), unit="s")


def guardar_json(docs, nombre):
    with open(os.path.join(OUT, nombre), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=1)
    print(f"  -> {nombre}: {len(docs):,} documentos")


def guardar_csv(df, nombre):
    df.to_csv(os.path.join(OUT, nombre), index=False, encoding="utf-8")
    print(f"  -> {nombre}: {len(df):,} filas")


# ---------------------------------------------------------------------------
# 1. VENTAS (datos reales de Kaggle, limpiados y enriquecidos)
# ---------------------------------------------------------------------------
print("[1/6] Limpiando ventas reales...")
csv = sorted(glob.glob(os.path.join(RAW, "*.csv")))[0]
df = pd.read_csv(csv, dtype={c: str for c in
                             ["order_id", "product_id", "category_id", "user_id"]})
print(f"  Filas originales: {len(df):,}")

df["fecha"] = pd.to_datetime(
    df["event_time"].astype(str).str.replace(" UTC", "", regex=False),
    errors="coerce")
df = df.dropna(subset=["fecha", "order_id", "product_id", "price"])
df = df[(df["fecha"].dt.year >= 2020) & (df["price"] > 0)].copy()  # quita fechas 1970 y precios 0

df["category_code"] = df["category_code"].fillna("otros.sin_categoria")
df["brand"] = df["brand"].fillna("generico")
df["categoria"] = df["category_code"].str.split(".").str[-1]
df["producto"] = (df["brand"].str.title() + " "
                  + df["categoria"].str.replace("_", " "))

# Clientes faltantes -> se asignan a un cliente existente
sin_cliente = df["user_id"].isna()
pool = df.loc[~sin_cliente, "user_id"].unique()
df.loc[sin_cliente, "user_id"] = rng.choice(pool, sin_cliente.sum())

# Canal, tienda y método de pago a nivel de orden (una orden = una tienda y un pago)
ordenes = df[["order_id"]].drop_duplicates().reset_index(drop=True)
m = len(ordenes)
fisica = rng.random(m) < 0.55
ordenes["canal"] = np.where(fisica, "Tienda física", "E-commerce")
ordenes["tienda"] = np.where(
    fisica, rng.choice(TIENDAS, m, p=[.25, .20, .25, .15, .15]), "E-commerce")
pago_fisica = rng.choice(["Tarjeta de crédito", "Tarjeta de débito", "Efectivo", "Yape", "Plin"],
                         m, p=[.30, .28, .20, .17, .05])
pago_online = rng.choice(["Tarjeta de crédito", "Tarjeta de débito", "Yape", "Plin", "Transferencia"],
                         m, p=[.45, .28, .15, .05, .07])
ordenes["metodo_pago"] = np.where(fisica, pago_fisica, pago_online)
df = df.merge(ordenes, on="order_id", how="left")

ventas = df.rename(columns={
    "order_id": "id_transaccion", "product_id": "id_producto",
    "brand": "marca", "price": "precio", "user_id": "id_cliente"})
ventas["cantidad"] = 1
ventas = ventas[["id_transaccion", "fecha", "id_producto", "producto", "categoria",
                 "marca", "precio", "cantidad", "tienda", "canal",
                 "metodo_pago", "id_cliente"]].sort_values("fecha")
print(f"  Filas tras limpieza: {len(ventas):,}")
guardar_csv(ventas, "ventas.csv")

F_INI, F_FIN = ventas["fecha"].min(), ventas["fecha"].max()
productos = (ventas[["id_producto", "producto", "categoria", "marca", "precio"]]
             .drop_duplicates("id_producto").reset_index(drop=True))
clientes_ids = ventas["id_cliente"].unique()

# ---------------------------------------------------------------------------
# 2. E-COMMERCE (clics / sesiones de navegación simuladas)
# ---------------------------------------------------------------------------
print("[2/6] Simulando clics de e-commerce...")
n = N_CLICKS
p = productos.iloc[rng.integers(0, len(productos), n)].reset_index(drop=True)
tiempo = rng.gamma(2.0, 60.0, n).round(1)                    # segundos, media ~120
prob_compra = np.clip(0.02 + tiempo * 0.0004, 0, 0.35)       # más tiempo -> más compra
compro = rng.random(n) < prob_compra
clicks = pd.DataFrame({
    "id_sesion": [f"S{i:07d}" for i in range(n)],
    "id_usuario": np.where(rng.random(n) < 0.7,
                           rng.choice(clientes_ids, n), "anonimo"),
    "fecha_hora": fechas_aleatorias(F_INI, F_FIN, n),
    "producto_visitado": p["producto"],
    "id_producto": p["id_producto"],
    "categoria": p["categoria"],
    "tiempo_navegacion_seg": tiempo,
    "paginas_vistas": rng.poisson(4, n) + 1,
    "dispositivo": rng.choice(["Móvil", "Escritorio", "Tablet"], n, p=[.62, .32, .06]),
    "origen_trafico": rng.choice(["Búsqueda orgánica", "Redes sociales", "Publicidad",
                                  "Email", "Directo"], n, p=[.30, .25, .20, .10, .15]),
    "agrego_carrito": compro | (rng.random(n) < 0.10),
    "compro": compro,
}).sort_values("fecha_hora")
guardar_csv(clicks, "ecommerce_clicks.csv")

# ---------------------------------------------------------------------------
# 3. REDES SOCIALES (comentarios simulados en JSON)
# ---------------------------------------------------------------------------
print("[3/6] Simulando comentarios de redes sociales...")
PLANTILLAS = {
    "positivo": [
        ("Me encantó mi nuevo {p}, llegó rapidísimo", ["envío rápido", "satisfecho"]),
        ("Excelente atención en {t}, el {p} funciona perfecto", ["buena atención", "calidad"]),
        ("Lo que compré de {m} en DataRetail es buenazo, 100% recomendado", ["recomendado", "calidad"]),
        ("Buen precio del {p} en el cyber, volveré a comprar", ["buen precio", "oferta"]),
    ],
    "negativo": [
        ("El delivery de mi {p} demoró más de una semana", ["demora", "delivery"]),
        ("Mi {p} llegó con fallas y soporte no responde", ["falla", "soporte"]),
        ("Precios más altos que la competencia en {c}", ["precio alto", "competencia"]),
        ("Pésima atención en {t}, nadie me ayudó", ["mala atención"]),
    ],
    "neutral": [
        ("¿Alguien sabe si el {p} tiene garantía de 2 años?", ["garantía", "consulta"]),
        ("¿Tienen stock de {m} en {t}?", ["stock", "consulta"]),
        ("Estoy comparando el {p} con otras marcas, ¿qué opinan?", ["comparación", "consulta"]),
    ],
}
sentimientos = rng.choice(["positivo", "negativo", "neutral"], N_COMENTARIOS, p=[.5, .3, .2])
redes_fechas = fechas_aleatorias(F_INI, F_FIN, N_COMENTARIOS)
comentarios = []
for i, sent in enumerate(sentimientos):
    prod = productos.iloc[int(rng.integers(0, len(productos)))]
    texto, claves = PLANTILLAS[sent][int(rng.integers(0, len(PLANTILLAS[sent])))]
    comentarios.append({
        "_id": f"C{i:06d}",
        "red_social": str(rng.choice(["Facebook", "Instagram", "TikTok", "X", "Google Reviews"],
                                     p=[.30, .25, .20, .10, .15])),
        "usuario": "@" + fake.user_name(),
        "fecha": fecha_mongo(redes_fechas[i]),
        "comentario": texto.format(p=prod["producto"], m=prod["marca"].title(),
                                   c=prod["categoria"], t=str(rng.choice(TIENDAS))),
        "palabras_clave": claves + [prod["categoria"], prod["marca"]],
        "producto_mencionado": prod["producto"],
        "sentimiento": str(sent),
        "likes": int(rng.poisson(15)),
    })
guardar_json(comentarios, "redes_sociales.json")

# ---------------------------------------------------------------------------
# 4. IoT (sensores por hora en cada tienda física)
# ---------------------------------------------------------------------------
print("[4/6] Simulando sensores IoT...")
horas = pd.date_range(F_INI.floor("D"), F_FIN.ceil("D"), freq="h", inclusive="left")
iot = pd.MultiIndex.from_product([TIENDAS, horas],
                                 names=["tienda", "fecha_hora"]).to_frame(index=False)
# Afluencia por hora del día (tiendas abren 9:00 y cierran 22:00)
perfil = np.array([0] * 9 + [8, 14, 20, 26, 30, 24, 22, 28, 35, 42, 45, 38, 20] + [0, 0])
factor_tienda = iot["tienda"].map(dict(zip(TIENDAS, [1.2, 1.0, 1.5, 1.3, 0.8]))).values
h = iot["fecha_hora"].dt.hour.values
finde = iot["fecha_hora"].dt.dayofweek.values >= 5
visitantes = rng.poisson(perfil[h] * factor_tienda * np.where(finde, 1.4, 1.0))
temperatura = (21 + 0.08 * visitantes + 1.5 * np.sin((h - 6) / 24 * 2 * np.pi)
               + rng.normal(0, 0.6, len(iot)))
falla_aire = rng.random(len(iot)) < 0.003            # anomalías para detectar en el análisis
temperatura = np.where(falla_aire, temperatura + rng.uniform(5, 9, len(iot)), temperatura)
iot["sensor_id"] = "IOT-" + iot["tienda"].map({t: f"{i + 1:02d}" for i, t in enumerate(TIENDAS)})
iot["clientes_detectados"] = visitantes
iot["temperatura_c"] = temperatura.round(1)
iot["humedad_pct"] = np.clip(rng.normal(65, 6, len(iot)), 40, 90).round(1)
iot["alerta_temperatura"] = iot["temperatura_c"] > 27
guardar_csv(iot, "iot_sensores.csv")

# ---------------------------------------------------------------------------
# 5. CLIENTES (documentos para MongoDB)
# ---------------------------------------------------------------------------
print("[5/6] Generando clientes para MongoDB...")
resumen = ventas.groupby("id_cliente").agg(
    num_compras=("id_transaccion", "nunique"),
    total_gastado=("precio", "sum"),
    primera_compra=("fecha", "min"),
    ultima_compra=("fecha", "max"))
favorita = (ventas.groupby(["id_cliente", "categoria"]).size().reset_index(name="n")
            .sort_values("n").drop_duplicates("id_cliente", keep="last")
            .set_index("id_cliente")["categoria"])
resumen["categoria_favorita"] = favorita
resumen["nivel"] = pd.qcut(resumen["total_gastado"].rank(method="first"), 4,
                           labels=["Bronce", "Plata", "Oro", "Platino"])
if len(resumen) > MAX_CLIENTES_MONGO:
    resumen = resumen.sample(MAX_CLIENTES_MONGO, random_state=SEMILLA)

clientes = []
for cid, r in resumen.iterrows():
    nombre, apellido = fake.first_name(), fake.last_name()
    dni_ficticio = f"{int(rng.integers(10_000_000, 99_999_999))}"
    registro = r["primera_compra"] - pd.Timedelta(days=int(rng.integers(1, 400)))
    clientes.append({
        "_id": str(cid),
        "nombre": nombre,
        "apellido": apellido,
        "email": fake.email(),
        "telefono": "9" + "".join(str(d) for d in rng.integers(0, 10, 8)),
        "dni_hash": hashlib.sha256(dni_ficticio.encode()).hexdigest(),  # seudonimizado
        "direccion": {"distrito": str(rng.choice(DISTRITOS)), "ciudad": "Lima"},
        "fecha_registro": fecha_mongo(registro),
        "nivel_fidelizacion": str(r["nivel"]),
        "consentimiento": {
            "tratamiento_datos": True,
            "marketing": bool(rng.random() < 0.6),
            "fecha": fecha_mongo(registro),
            "version_politica": "v1.0",
        },
        "resumen_compras": {
            "num_compras": int(r["num_compras"]),
            "total_gastado": round(float(r["total_gastado"]), 2),
            "categoria_favorita": str(r["categoria_favorita"]),
            "ultima_compra": fecha_mongo(r["ultima_compra"]),
        },
    })
guardar_json(clientes, "clientes.json")

# ---------------------------------------------------------------------------
# 6. VENTAS MUESTRA (documentos anidados para MongoDB)
# ---------------------------------------------------------------------------
print("[6/6] Generando muestra de ventas para MongoDB...")
ids_mongo = set(resumen.index)
muestra = ventas[ventas["id_cliente"].isin(ids_mongo)]
muestra = muestra.sample(min(N_VENTAS_MONGO, len(muestra)), random_state=SEMILLA)
ventas_docs = [{
    "_id": f"V{i:06d}",
    "id_transaccion": v.id_transaccion,
    "fecha": fecha_mongo(v.fecha),
    "cliente_id": v.id_cliente,
    "canal": v.canal,
    "tienda": v.tienda,
    "metodo_pago": v.metodo_pago,
    "producto": {"id": v.id_producto, "nombre": v.producto,
                 "categoria": v.categoria, "marca": v.marca},
    "precio": round(float(v.precio), 2),
    "cantidad": int(v.cantidad),
} for i, v in enumerate(muestra.itertuples(index=False))]
guardar_json(ventas_docs, "ventas_muestra.json")

print("\nListo. Archivos en:", OUT)
