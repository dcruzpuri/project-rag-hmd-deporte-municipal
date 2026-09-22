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

Valores aplicados por `src/chunk.py`:

- `CHUNK_SIZE`: 1000
- `CHUNK_OVERLAP`: 100
- Chunks resultantes: 16214
- Tamaño medio: 660 tokens
- Rango: 27 – 3397 tokens

### 2.1. Estrategia de troceado

- **Splitter**: `RecursiveCharacterTextSplitter`, el "Level 2 – Recursive Character" del tutorial [*Five levels of text splitting*](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb): se corta primero en límites semánticos y solo después en espacios/caracteres.
- **Orden de separadores**: párrafo (`\n\n`) → salto de línea (`\n`) → frase (`. `) → espacio → carácter. Un corte cae en el límite más fino posible, nunca en medio de un párrafo si cabe.
- **1000/100**: límite elegido entre coste de embedding (fragmentos demasiado largos diluyen el vector en un modelo de 4B) y contexto útil (fragmentos cortos pierden el "cómo/por qué" que rodea al dato). El overlap de 10 % hace que la zona de corte exista en los dos chunks adyacentes: una respuesta partida a mitad de frase sigue siendo recuperable y legible.
- **Metadata heredada**: cada chunk copia la metadata del documento padre (`source`, `file_hash`, `csv_kind`, `dedup_policy`, `entity_key`/`group_key`, tags del TAG) y añade su propio `chunk_index` y `chunk_size`. Es el puente que permite deduplicar por clave exacta y auditar en ChromaDB.
- El valor máximo (3397) corresponde a entidades `no_chunk` íntegras: un `max > CHUNK_SIZE` esperado, no un fallo del splitter.

### 2.2. Estrategia por tipo de documento

El chunking no es uniforme, y el criterio es **el contenido, no la extensión**: los CSV se deciden en `src/csv_advisor.py` **antes** de trocear (receta de fuente conocida `RECETAS_CONOCIDAS` → heurística de perfil estructural → fallback conservador), los `.txt`/`.md` pasan primero por la detección de tabulares de `src/load.py` (sección 2.3), y todo deposita un `chunking_hint` que consume `src/chunk.py`:

| Tipo de documento | Origen | `chunking_hint` | Qué se trocea | Por qué |
|---|---|---|---|---|
| PDF / TXT / MD (texto plano) | páginas `PyPDFLoader` / `TextLoader` | `normal_chunk` (default) | Splitter recursivo 1000/100 | Texto de longitud variable (reglamentos, infografías); el fragmento debe caber en la ventana de embedding del modelo |
| Tabular detectado en `.txt`/`.md` | CSV/TSV disfrazado que pasa `_es_tabular()` (p. ej. `211549-0-juegos-deportivos-actual.txt`) | `normal_chunk` por fila (`tabular_txt`) | 1 fila = 1 documento; el splitter solo actúa si la fila supera el CHUNK_SIZE | El advisor NO se usa aquí a propósito: un TSV con campo repetido se clasificaría como entidad y la dedup `exact_key` colapsaría todas las filas (fallo 4) |
| Fila de entidad | CSV con ID estable: 200186 (polideportivos), 200215 (instalaciones), 210227 (piscinas), 300390 (áreas), 212504 (agenda) | `no_chunk` | No se trocea: 1 fila = 1 chunk | Una fila es una entidad indivisible (nombre + distrito + dirección…). Trocear separaría el ID de sus atributos y rompería la dedup `exact_key` que necesita ese ID |
| Grupo de hechos | CSV sin ID, con medidas numéricas: 300085 (abonos), 300097 (descuentos) | `light_chunk` | Sin trocear si el grupo ≤ 1000; recursivo si lo supera | El advisor agrupa las filas por dimensiones (mes × centro × tipo) en un documento por grupo con desglose. Miles de casi-duplicados genéricos saturarían el espacio vectorial indexando un "resumen + desglose" por vez |
| Tabla textual / CSV desconocido | Cualquier otro CSV | `normal_chunk` | Fila por documento troceado | Fallback conservador: la fila viaja sin troceado si es atómica |

### 2.3. Detección de tabulares en archivos de texto (cambio reciente)

Los `.txt`/`.md` ya no se cargan ciegamente como texto plano: antes de trocear, `src/load.py` comprueba si el archivo es **tabular disfrazado** (CSV/TSV con extensión de texto) y, en ese caso, lo procesa por el mismo pipeline que los CSV (1 fila = 1 documento).

- **Detección** (`_es_tabular`): reutiliza `leer_filas_csv` del pipeline de CSVs (encoding `utf-8-sig` → `cp1252` → `latin-1` y delimitador `;`/`,`/`\t` detectado por columna estable) y muestrea solo las primeras 50 filas: el archivo es tabular si hay **dos o más columnas y por encima de 50 filas**, además de conservar un número de columnas estable. Cualquier fallo de lectura se trata como "no tabular" (no rompe la carga de docs.).
- **Clasificación forzada**: un `CsvAdvice` con `csv_kind=tabular_txt`, `treatment=row_as_doc` y `dedup_policy=exact_only` — es decir, **sin pasar por el advisor de CSVs**. Razón: un TSV con un campo que parece clave (p. ej. `Codigo_temporada`, repetido en muchas filas) se clasificaría `entity_doc` + `exact_key`, y la deduplicación por clave colapsaría todas las filas en un único documento.
- **Evidencia** (auditoría de dedup sobre `211549-0-juegos-deportivos-actual.txt`, fallo 4): sin la detección, el archivo se cargaba como 1 documento → 1892 chunks erróneos → 476 tras deduplicación semántica agresiva (pérdida de ~1.400 chunks). Con la detección: **8769 filas = 8769 documentos** y 0 pérdidas por dedup semántica.
- **Metadata**: cada documento lleva `doc_id` (ordinal dentro del archivo), que con `source` + `chunk_index` hace inequívoca cada fila para la auditoría de deduplicación.

### 2.4. Validación

- **Semántica** (`scripts/eval_coherencia_chunks.py`): sobre un texto de dominio, compara la similitud coseno media de pares adyacentes con pares aleatorios; margen > 0.05 → el overlap mantiene coherencia temática; margen prácticamente 0 (muy cercano) → cortes arbitrarios.

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

Se ha probado la misma pregunta con tres valores de K sobre el índice actual (16214 chunks, 2560 dims).

**Pregunta de referencia:** *"¿Qué piscinas municipales hay en Madrid?"*

| TOP_K | Chunk #1 (fuente / distancia) | ¿Aparece PiscinasAireLibre2026.pdf? | Respuesta final | Observaciones |
|---|---|---|---|---|
| 1 | PiscinasAireLibre2026.pdf / 0.4310 | Sí (chunk #1) | Correcta, pero parcial | Solo hay un fragmento; Gemini responde con menos detalle |
| 3 | PiscinasAireLibre2026.pdf / 0.4310 | Sí (chunks #1 y #2) | Buena | Equilibrio entre precisión y cobertura; se evita ruido |
| 5 | PiscinasAireLibre2026.pdf / 0.4310 | Sí (chunks #1 y #2) | Completa | Aparecen fuentes adicionales (`reglamento_instalaciones.pdf`, `200186-0-polideportivos.csv`) que enriquecen la respuesta |

### 4.1. Conclusiones

- **K=1**: suficiente para preguntas muy específicas cuya respuesta está en un solo fragmento. Riesgo: se pierde información complementaria (por ejemplo, distrito o dirección).
- **K=3**: **punto dulce**. Devuelve toda la información relevante sin introducir ruido perceptible.
- **K=5**: útil cuando la pregunta requiere cruzar varios documentos (por ejemplo, precio + ubicación + horario). Con corpus desbalanceados puede introducir algún chunk del CSV de descuentos, pero el prompt restrictivo evita que Gemini se apoye en él si no responde a la pregunta.

### 4.2. Valor por defecto elegido

`TOP_K=5` como valor por defecto en `config.py` porque:

- El corpus es heterogéneo (PDF + CSV + TXT) y las respuestas suelen estar repartidas.
- El coste en tokens de Gemini es marginal para +2 chunks.
- El prompt con grounding filtra bien el ruido: si un chunk no aporta, Gemini simplemente lo ignora.

Se puede sobrescribir en cada consulta con `--top-k N` en la CLI o `top_k=N` en `responder()`, para pruebas específicas.

---

## 5. Éxito in-corpus

**Pregunta:** ¿Qué piscinas municipales hay en Madrid?

**Respuesta:** El sistema recuperó correctamente 18 piscinas con sus fuentes documentadas.

Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 398/398 [00:05<00:00, 71.01it/s]
2026-09-20 19:13:14 [EMBED]   huggingface: 1 textos en una única llamada (batch_size=32)
2026-09-20 19:13:14 [EMBED]   huggingface: 1 vectores en 0.7 s
Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
[RAG] [WARN] Error transitorio de Gemini (1/3): reintentando en 2s
2026-09-20 19:13:30 [RAG] [INFO] {"pregunta": "¿Qué piscinas municipales hay en Madrid?", "top_k": 5, "n_chunks": 5, "modelo": "google:gemini-3.6-flash", "abstained": false, "t_retrieval": 14.9789, "t_generation": 15.0089}

=== Respuesta ===
Según la información del documento `PiscinasAireLibre2026.pdf` (y también referenciada en `200186-0-polideportivos.csv`), las piscinas municipales al aire libre son:

* **Peñuelas** (Distrito Arganzuela)
* **Mistral** / **Centro Deportivo Municipal Mistral** (Distrito Barajas)
* **Blanca Fernández Ochoa** (Distrito Carabanchel)
* **Concepción** (Distrito Ciudad Lineal)
* **Santa Ana** (Distrito Fuencarral - El Pardo)
* **Vicente del Bosque** (Distrito Fuencarral - El Pardo)
* **Luis Aragonés** (Distrito Hortaleza)
* **Hortaleza** (Distrito Hortaleza)
* **Aluche** (Distrito Latina)
* **Casa de Campo** (Distrito Moncloa - Aravaca)
* **San Blas** (Distrito San Blas - Canillejas)
* **Paseo de la Dirección** (Distrito Tetuán)
* **Moscardó** (Distrito Usera)
* **Orcasitas** (Distrito Usera)
* **San Fermín** (Distrito Usera)
* **Margot Moles** (Distrito Vicálvaro)
* **Cerro Almodóvar** (Distrito Villa de Vallecas)
* **Plata y Castañar** (Distrito Villaverde)

=== Fuentes ===
  - PiscinasAireLibre2026.pdf
  - reglamento_instalaciones.pdf
  - 200186-0-polideportivos.csv
  - 200215-0-instalaciones-deportivas.csv

=== Métricas ===
  retrieval: 14.9789
  generation: 15.0089
  top_k: 5
  n_chunks: 5
  model: gemini-3.6-flash

---

## 6. Abstención out-of-corpus

### 6.1. Caso 1: pregunta claramente fuera de dominio

**Pregunta:**

> ¿Cuál es la capital de Francia?

**Comando ejecutado:**

Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 398/398 [00:03<00:00, 100.00it/s]
2026-09-20 19:16:20 [EMBED]   huggingface: 1 textos en una única llamada (batch_size=32)
2026-09-20 19:16:20 [EMBED]   huggingface: 1 vectores en 0.5 s
2026-09-20 19:16:21 [RAG] [INFO] Sin evidencia suficiente para: '¿Cuál es la capital de Francia?' (top-k=5, chunks=5)

=== Respuesta ===
No dispongo de esa información en los documentos proporcionados.

=== Fuentes ===
  - 300097-0-deportes-descuentos.csv

[INFO] El sistema se abstuvo: no hay evidencia suficiente en el corpus.

=== Métricas ===
  retrieval: 11.703
  top_k: 5
  n_chunks: 0

---

## 7. Fallos conocidos

**Fallo 1 — Abstención por corpus incompleto**
Pregunta: ¿Qué instalaciones deportivas hay en Chamberí?
El sistema se abstiene porque los chunks no mencionan "Chamberí" explícitamente. Es un límite del corpus, no del RAG.

**Fallo 2 — Abstención en descuentos**
Pregunta: ¿Qué descuentos hay para abonados?
El CSV de descuentos contiene hechos pero no documenta explícitamente qué descuentos existen. El sistema se abstiene correctamente.

**Fallo 3 — `Tarifas_deportivas.pdf`**
El archivo acaba con 0 chunks. Posible causa: deduplicación agresiva, clave repetida o problema de carga.

- **Investigación:** [**FORTALEZA**] Se revisa el curado del corpus y se encuentra que el sistema de deduplicación discrimina el documento, dado que existe uno análogo con fecha más reciente (`PreciosPublicos2026.pdf`) que es casi idéntico y es el que toma como un único chunk para embedding. 

**Fallo 4 — Deduplicación agresiva sobre `211549-0-juegos-deportivos-actual.txt`**
El archivo pierde entorno a los 1.400 chunks por deduplicación agresiva al tomar el archivo entero como un único documento aparentemente colapsando todas las filas en una. Requiere investigación.
| fuente | docs | chunks pre | chunks post (dedup) |
|---|---|---|---|
| 200215-0-instalaciones-deportivas.csv | 607 | 607 | 607 |
| 20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf | 3 | 3 | 3 |
| **211549-0-juegos-deportivos-actual.txt** | 1 | 1892 | 476 |

- **Investigación:** El sistema de ingesta toma el archivo, que es de naturaleza tabular, como un único documento del que genera casi 1.900 chunks erróneamente. Se modifica la ingesta para que sea detectado en la carga de documentos (`load.py`) como archivo tabular, haciéndolo pasar por el mismo pipeline reaprovechado de los archivos CSV (ahora también trata TSV), lo que obtiene un grado mucho más acorde de chunking, a tenor de las líneas que contiene a modo de registro (1 fila = 1 documento), evitando caer por tanto en la deduplicación semántica:
| fuente | docs | chunks pre | chunks post (dedup) |
|---|---|---|---|
| 200215-0-instalaciones-deportivas.csv | 607 | 607 | 607 |
| 20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf | 3 | 3 | 3 |
| **211549-0-juegos-deportivos-actual.txt** | 8769 | 8769 | 8769 |


---

## 8. Siguientes pasos

[ ]Ampliar el corpus con documentos más descriptivos para reducir abstenciones por corpus incompleto.  
[✓] Investigar el fallo de `Tarifas_deportivas.pdf`.  
[✓] Investigar la deduplicación agresiva sobre `211549-0-juegos-deportivos-actual.txt`.  
[ ] Probar con modelo de embeddings de mayor dimensión.  
[ ] Evaluar con las 22 preguntas del dataset de evaluación.  
