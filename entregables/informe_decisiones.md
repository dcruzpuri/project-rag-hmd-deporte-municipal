# Informe de decisiones — RAG Deporte Municipal Madrid

## 1. Corpus utilizado

| Archivo | Formato | Contenido |
|---|---|---|
| 200186-0-polideportivos.csv | CSV | Centros deportivos municipales |  
| 200215-0-instalaciones-deportivas.csv | CSV | Instalaciones deportivas |
| 210227-0-piscinas-publicas.csv | CSV | Piscinas públicas municipales |
| 212504-0-agenda-actividades-deportes.csv | CSV | Agenda de actividades deportivas |
| 300085-0-deportes_abonos.csv | CSV | Abonos deportivos |
| 300097-0-deportes-descuentos.csv | CSV | Descuentos deportivos |
| 300390-0-areas-deportivas.csv | CSV | Áreas deportivas |

Total: 7756 documentos cargados, 16214 chunks generados.

---

## 2. Decisiones de chunking

- `CHUNK_SIZE`: 1000
- `CHUNK_OVERLAP`: 100
- Chunks resultantes: 16214
- Tamaño medio: 660 tokens
- Rango: 27 – 3397 tokens

@Héctor — explicar decisiones de chunking y estrategia por tipo de documento.

---

## 3. Experimento con diferentes configuraciones

### 3.1. Configuración base del índice

| Parámetro | Valor | Motivo |
|---|---|---|
| `EMBED_PROVIDER` | huggingface | Modelo local, sin coste de API, sin límite de cuota |
| `HF_EMBED_MODEL` | Qwen/Qwen3-Embedding-4B | 2560 dims, buen rendimiento en español; acordado con el equipo |
| `EMBED_DIM` | 2560 | Dim real del modelo, sin recorte |
| `CHUNK_SIZE` | 1000 | Equilibrio entre precisión y contexto |
| `CHUNK_OVERLAP` | 100 | 10 % del chunk_size, evita cortar ideas |
| `TAG_SCORING_DEDUP` | false | Desactivado para acelerar pruebas; ver sección 7 |
| `DEDUP_UMBRAL` | 0.93 | Valor por defecto del equipo |

- **Chunks indexados**: 16214
- **Dim del índice**: 2560
- **Tiempo de indexación**: ~15 min (RTX 4070 Ti SUPER, `TAG_SCORING_DEDUP=false`)

### 3.2. Por qué no se ha hecho un barrido de CHUNK_SIZE / CHUNK_OVERLAP

El offline (ingesta + chunking + embeddings + indexación) lo controla la parte de Héctor. Reindexar con distintos valores de `CHUNK_SIZE` implicaría volver a generar el índice completo (~15 min cada vez) y regenerar `output/embeddings.json`, lo que añade ~1 GB de disco por ejecución.

**Decisión del equipo**: mantener fijo el chunking en `1000/100` para las pruebas de retrieval, y centrar el experimento en `TOP_K` (sección 4), que es lo que sí cambia el comportamiento del retrieval sin necesidad de reindexar.

### 3.3. Observación sobre el corpus

El corpus tiene un desbalance claro: el CSV de descuentos (`300097-0-deportes-descuentos.csv`) aporta miles de chunks del tipo *"resumen del dataset…"* con estructura muy repetitiva. Esto provoca que el espacio vectorial tenga una densidad alta en esa zona y que ciertas preguntas devuelvan chunks de ese CSV con distancias medias (~0.46) aunque no respondan a la pregunta. Es un límite del corpus, no del retrieval.

---

## 4. Comparación de TOP_K

| TOP_K | Observaciones |
|---|---|
| 1 | TODO |
| 3 | TODO |
| 5 | TODO |

---

## 5. Éxito in-corpus

**Pregunta:** ¿Qué piscinas municipales hay en Madrid?

**Respuesta:** El sistema recuperó correctamente 18 piscinas con sus fuentes documentadas.

**Distancias de retrieval:** 0.43 – 0.53

---

## 6. Abstención out-of-corpus

**Pregunta:** ¿A qué hora juega el Madrid hoy?

**Respuesta del sistema:** "No dispongo de esa información en los documentos proporcionados."

**Motivo:** La pregunta no tiene relación con el corpus de Deporte Municipal Madrid.

---

## 7. Fallos conocidos

**Fallo 1 — Abstención por corpus incompleto**
Pregunta: ¿Qué instalaciones deportivas hay en Chamberí?
El sistema se abstiene porque los chunks no mencionan "Chamberí" explícitamente. Es un límite del corpus, no del RAG.

**Fallo 2 — Abstención en descuentos**
Pregunta: ¿Qué descuentos hay para abonados?
El CSV de descuentos contiene hechos pero no documenta explícitamente qué descuentos existen. El sistema se abstiene correctamente.

**Fallo 3 — Tarifas_deportivas.pdf**
El archivo acaba con 0 chunks. Posible causa: deduplicación agresiva, clave repetida o problema de carga. Pendiente de investigar.

---

## 8. Siguientes pasos

- Ampliar el corpus con documentos más descriptivos para reducir abstenciones por corpus incompleto
- Investigar el fallo de `Tarifas_deportivas.pdf`
- Probar con modelo de embeddings de mayor dimensión
- Evaluar con las 22 preguntas del dataset de evaluación