// ============================================================================
// 05_mongodb.js  -  Base de datos NoSQL de DataRetail en MongoDB
// Ejecutar:  docker exec -it mongodb mongosh --quiet /work/05_mongodb.js
// (o pegar cada bloque en la pestaña _MONGOSH de Compass)
// ============================================================================
db = db.getSiblingDB("dataretail");

function titulo(t) { print("\n" + "=".repeat(70) + "\n " + t + "\n" + "=".repeat(70)); }
function mostrar(cursor) { cursor.toArray().forEach(d => printjson(d)); }

titulo("0. COLECCIONES Y NÚMERO DE DOCUMENTOS");
printjson({
  clientes: db.clientes.countDocuments(),
  ventas: db.ventas.countDocuments(),
  comentarios_redes: db.comentarios_redes.countDocuments()
});

// ----------------------------------------------------------------------------
titulo("1. VALIDACIÓN DE ESQUEMA (calidad y cumplimiento de datos)");
// Todo cliente debe tener email válido y un registro de consentimiento (Ley 29733)
db.runCommand({
  collMod: "clientes",
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["nombre", "email", "consentimiento"],
      properties: {
        email: { bsonType: "string", pattern: "^.+@.+\\..+$" },
        nivel_fidelizacion: { enum: ["Bronce", "Plata", "Oro", "Platino"] },
        consentimiento: {
          bsonType: "object",
          required: ["tratamiento_datos", "fecha", "version_politica"],
          properties: { tratamiento_datos: { bsonType: "bool" }, marketing: { bsonType: "bool" } }
        }
      }
    }
  },
  validationLevel: "strict",
  validationAction: "error"
});
print("Validador aplicado a 'clientes'. Prueba: insertar un cliente SIN consentimiento...");
try {
  db.clientes.insertOne({ _id: "prueba", nombre: "Test", email: "test@correo.com" });
} catch (e) {
  print("  RECHAZADO por el validador (correcto): " + e.errmsg);
}

// ----------------------------------------------------------------------------
titulo("2. ÍNDICES (aceleran las consultas frecuentes)");
db.ventas.createIndex({ fecha: 1 });
db.ventas.createIndex({ cliente_id: 1 });
db.ventas.createIndex({ "producto.categoria": 1, precio: -1 });
db.clientes.createIndex({ nivel_fidelizacion: 1, "direccion.distrito": 1 });
db.comentarios_redes.createIndex({ sentimiento: 1, red_social: 1 });
db.comentarios_redes.createIndex({ comentario: "text", palabras_clave: "text" },
                                 { default_language: "spanish" });
["ventas", "clientes", "comentarios_redes"].forEach(c =>
  print(c + ": " + db[c].getIndexes().map(i => i.name).join(", ")));

titulo("   Prueba de rendimiento del índice (explain)");
const st = db.ventas.find({ "producto.categoria": "smartphone" }).explain("executionStats").executionStats;
printjson({ documentos_devueltos: st.nReturned, documentos_examinados: st.totalDocsExamined,
            claves_indice_examinadas: st.totalKeysExamined, tiempo_ms: st.executionTimeMillis });

// ----------------------------------------------------------------------------
titulo("3. CONSULTA: clientes Platino que aceptaron marketing (para campaña)");
mostrar(db.clientes.find(
  { nivel_fidelizacion: "Platino", "consentimiento.marketing": true },
  { nombre: 1, apellido: 1, "direccion.distrito": 1, "resumen_compras.total_gastado": 1, _id: 0 }
).sort({ "resumen_compras.total_gastado": -1 }).limit(5));

// ----------------------------------------------------------------------------
titulo("4. AGREGACIÓN: ingresos por categoría (top 10)");
mostrar(db.ventas.aggregate([
  { $group: { _id: "$producto.categoria", ingresos: { $sum: "$precio" }, ventas: { $sum: 1 } } },
  { $project: { categoria: "$_id", _id: 0, ventas: 1, ingresos: { $round: ["$ingresos", 2] } } },
  { $sort: { ingresos: -1 } }, { $limit: 10 }
]));

titulo("5. AGREGACIÓN: ventas por mes");
mostrar(db.ventas.aggregate([
  { $group: { _id: { $dateToString: { format: "%Y-%m", date: "$fecha" } },
              ingresos: { $sum: "$precio" }, ventas: { $sum: 1 } } },
  { $project: { mes: "$_id", _id: 0, ventas: 1, ingresos: { $round: ["$ingresos", 2] } } },
  { $sort: { mes: 1 } }
]));

titulo("6. AGREGACIÓN: métodos de pago por canal");
mostrar(db.ventas.aggregate([
  { $group: { _id: { canal: "$canal", pago: "$metodo_pago" }, transacciones: { $sum: 1 } } },
  { $sort: { "_id.canal": 1, transacciones: -1 } }
]));

titulo("7. $lookup (JOIN): top 5 clientes por gasto con sus datos");
mostrar(db.ventas.aggregate([
  { $group: { _id: "$cliente_id", gasto: { $sum: "$precio" }, compras: { $sum: 1 } } },
  { $sort: { gasto: -1 } }, { $limit: 5 },
  { $lookup: { from: "clientes", localField: "_id", foreignField: "_id", as: "cliente" } },
  { $unwind: "$cliente" },
  { $project: { _id: 0, cliente: { $concat: ["$cliente.nombre", " ", "$cliente.apellido"] },
                nivel: "$cliente.nivel_fidelizacion", distrito: "$cliente.direccion.distrito",
                compras: 1, gasto: { $round: ["$gasto", 2] } } }
]));

titulo("8. AGREGACIÓN: clientes por nivel y % que acepta marketing");
mostrar(db.clientes.aggregate([
  { $group: { _id: "$nivel_fidelizacion", clientes: { $sum: 1 },
              aceptan_marketing: { $sum: { $cond: ["$consentimiento.marketing", 1, 0] } },
              gasto_promedio: { $avg: "$resumen_compras.total_gastado" } } },
  { $project: { nivel: "$_id", _id: 0, clientes: 1, gasto_promedio: { $round: ["$gasto_promedio", 2] },
                pct_marketing: { $round: [{ $multiply: [{ $divide: ["$aceptan_marketing", "$clientes"] }, 100] }, 1] } } },
  { $sort: { gasto_promedio: -1 } }
]));

// ----------------------------------------------------------------------------
titulo("9. REDES SOCIALES: sentimiento por red social");
mostrar(db.comentarios_redes.aggregate([
  { $group: { _id: { red: "$red_social", sentimiento: "$sentimiento" }, total: { $sum: 1 } } },
  { $group: { _id: "$_id.red", detalle: { $push: { k: "$_id.sentimiento", v: "$total" } } } },
  { $project: { red_social: "$_id", _id: 0, conteo: { $arrayToObject: "$detalle" } } },
  { $sort: { red_social: 1 } }
]));

titulo("10. REDES SOCIALES: palabras clave más frecuentes en quejas");
mostrar(db.comentarios_redes.aggregate([
  { $match: { sentimiento: "negativo" } },
  { $unwind: "$palabras_clave" },
  { $group: { _id: "$palabras_clave", menciones: { $sum: 1 } } },
  { $sort: { menciones: -1 } }, { $limit: 8 }
]));

titulo("11. BÚSQUEDA DE TEXTO: comentarios sobre 'demora' o 'falla'");
mostrar(db.comentarios_redes.find(
  { $text: { $search: "demora falla" } },
  { _id: 0, red_social: 1, comentario: 1, sentimiento: 1, score: { $meta: "textScore" } }
).sort({ score: { $meta: "textScore" } }).limit(5));

print("\nListo.");
