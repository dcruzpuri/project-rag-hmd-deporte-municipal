# RAG Deporte Municipal Madrid

Asistente conversacional que responde preguntas sobre el deporte municipal de Madrid usando Retrieval-Augmented Generation (RAG). Los datos vienen del portal de datos abiertos del Ayuntamiento de Madrid.

## Equipo

- **Héctor** — corpus, pipeline de indexación, ChromaDB
- **Miguel** — retrieval, generación, orquestación (`responder()`)
- **David** — interfaz Streamlit, evaluación, documentación

## Requisitos

- Python 3.10+
- Ollama instalado y corriendo (`ollama serve`)
- API key de Google Gemini
<<<<<<< HEAD
# Generación Aumentada por Recuperación (RAG) para el Deporte Municipal del Ayuntamiento de Madrid
=======
# project-rag-hmd-deporte-municipal

La interfaz web permite interactuar con el sistema RAG de forma visual.

## Paso 0 — Instalación y arranque
# Retrieval Augmented Generation (RAG) para Deporte Municipal del Ayuntamiento de Madrid
>>>>>>> origin/develop

Este proyecto implementa un sistema RAG (Retrieval Augmented Generation, generación aumentada por recuperación) que responde preguntas a partir del contenido recuperado de los documentos sobre deporte municipal del Ayuntamiento de Madrid: tarifas, abonos, reservas, horarios e instalaciones.

La carga inicial incluye un corpus real de documentos en formato PDF, CSV y TXT procedente de la sede de datos abiertos del Ayuntamiento de Madrid. A continuación, el sistema trocea el corpus, genera los embeddings, etiqueta las fuentes con un modelo de lenguaje, y evalúa y mejora la calidad semántica mediante puntuación y deduplicación (el bloque TSD, descrito más adelante, es opcional), y guarda el resultado en la base de datos vectorial **ChromaDB** para la recuperación semántica. Se trata de un pipeline **offline** completo; la fase online (recuperación, generación e interfaz de usuario) consta en el [mapa de ruta](#roadmap).

Las fases del pipeline se encadenan en este orden: preverificación (PREFLIGHT), carga (LOAD), limpieza (CLEAN), etiquetado (TAG, mediante un modelo de lenguaje), troceado (CHUNK), vectorización (EMBED), puntuación (SCORING), deduplicación (DEDUP), indexación (INDEX) y generación del informe (INFORME).

## Características principales

- **Preverificación de disponibilidad**: el pipeline comprueba que el modelo de la constante `EMBED_MODEL` está disponible en el proveedor indicado por la constante `EMBED_PROVIDER` (el punto de servicio `/api/tags` de Ollama, la comprobación `repo_exists` de HuggingFace o el listado de modelos de Google) y **se detiene** si no lo está. Cuando la verificación en línea no es posible (red cortada o ausencia de clave API), la comprobación se degrada a la memoria caché offline (las constantes `EMBED_DIM_MAX_*`) y se emite un aviso, en lugar de interrumpir el proceso.
- **Proveedor múltiple sin modificar código**: se elige entre `ollama`, `huggingface` y `google` mediante variables de entorno, a través de tres interruptores independientes: `EMBED_PROVIDER`, `TAG_PROVIDER` y `GEN_PROVIDER`. El etiquetado hereda el proveedor de generación por defecto cuando no se define un valor, en todos los casos limitado a estos tres proveedores.
- **Bloque TSD** (etiquetado, puntuación y deduplicación, opcional): en primer lugar, etiqueta cada fuente con un modelo de lenguaje mediante una taxonomía cerrada; a continuación, asigna a cada chunk una puntuación semántica que combina la relevancia adjudicada por el modelo de lenguaje con la centralidad y la no redundancia (ambas medidas sobre los vectores mediante FAISS) y la autoridad (un ajuste fino según el tipo de fuente); por último, aplica una deduplicación de tipo greedy: ordena los chunks por puntuación y conserva cada uno únicamente cuando no es demasiado parecido a los que ya se han conservado (el umbral corresponde a la constante `DEDUP_UMBRAL`). El proceso respeta la política específica de los archivos CSV, de modo que el índice contiene más información útil y menos contenido redundante.
- **Tratamiento diferenciado de los archivos CSV**: el módulo `csv_advisor` analiza y clasifica cada archivo (entidad con identificador estable, tabla de hechos, contenido textual o archivo desconocido) y decide cómo convertirlo en documento para indexación y cómo aplicar la deduplicación. La deduplicación de los archivos CSV **nunca es semántica**; se realiza por clave exacta. Ver el [detalle](docs/ANEXO_FUNC.md#3-capa-csv-clasificación-diferencial).
- **Dimensión garantizada**: la dimensión final de todos los vectores del índice es siempre el mínimo entre la dimensión generada y la constante `EMBED_DIM`. Cuando el proveedor lo permite, se solicita esa dimensión en el servidor (en Ollama mediante el parámetro `dimensions` y en HuggingFace mediante `truncate_dim`); después, el cliente recorta el prefijo del vector si es necesario y lo renormaliza. Si el valor de `EMBED_DIM` es menor que la dimensión máxima del modelo, se emite un mensaje informativo (tipo INFO); si es mayor que la dimensión generada, se emite un aviso (tipo AVISO) y se usa la dimensión generada, dejando constancia de ambas dimensiones en el registro, lo que **evita incompatibilidades en ChromaDB**.
- **Índice regenerable** en ChromaDB persistente (métrica coseno): cada ejecución puede reconstruir el índice a partir del corpus, sin depender de inserciones anteriores. Los vectores se insertan por lotes para controlar el uso de memoria y mejorar el rendimiento. Al finalizar, el pipeline vuelve a consultar los identificadores insertados y compara el resultado con el esperado, de modo que detecta inserciones incompletas o reindexaciones parciales antes de dar el proceso por válido.
- **Informe de indexación** en formato markdown al terminar cada ejecución: parámetros aplicados, resultados de la preverificación, tiempos por fase, datos del troceado, estado del índice, métricas del bloque TSD y **señales automáticas** con la acción recomendada en cada caso. El nombre del archivo sigue el patrón `output/informe_index_GUID8_YYYYMMDD_HHMM.md`, donde GUID8 son los ocho primeros caracteres del identificador único de la base de datos de ChromaDB, y YYYYMMDD y HHMM son la fecha y la hora locales de la ejecución.
- **Registro en consola** de cada fase, además del diccionario de métricas que devuelve la función `ejecutar_pipeline`, listo para su evaluación.

Preguntas de ejemplo; el corpus debe responder a las 18 preguntas de [`queries/preguntas.json`](queries/preguntas.json), entre las que figuran:

- ¿Cuánto cuesta el abono de piscina?
- ¿Hay descuentos para menores, mayores, estudiantes o personas en paro?
- ¿Puedo reservar siendo no empadronado?
- ¿Qué pasa si cancelo una reserva con coste?

## Estructura del proyecto

La raíz del proyecto contiene los elementos siguientes:

- `README.md`: descripción general del proyecto (este documento).
- `requirements.txt`: dependencias del proyecto (cargadores de LangChain, ChromaDB, cliente HTTP para Ollama, Google Gemini, sentence-transformers, FAISS, pandas, Streamlit y utilidades).
- `.env.example`: plantilla de configuración; se copia a `.env` y se edita (incluye los valores por defecto de cada variable).
- `config.py`: resuelve todas las variables del archivo `.env` como constantes de código, con sus valores por defecto.
- `src/pipeline.py`: orquestador del pipeline; encadena todas las fases (de la preverificación hasta el informe) y devuelve el diccionario de métricas.
- `src/load.py`: fase LOAD; carga archivos PDF, TXT, MD y CSV (un archivo o una carpeta) y los convierte en documentos.
- `src/clean.py`: fase CLEAN; normaliza saltos de línea, espacios, carácter BOM y caracteres de control.
- `src/chunk.py`: fase CHUNK; troceado con el divisor recursivo de texto por caracteres (separadores de párrafo, frase y letra).
- `src/embed.py`: fase EMBED; vectorización multi-proveedor (Ollama, HuggingFace, Google) y preverificación de disponibilidad del modelo.
- `src/index.py`: fase INDEX; inserción en ChromaDB persistente con métrica coseno y verificación de los identificadores.
- `src/informe.py`: genera el archivo del informe de indexación descrito en la sección de características.
- `src/csv_advisor.py`: clasifica cada archivo CSV y decide su tratamiento y la política de deduplicación.
- `src/csv_transform.py`: materializa el consejo del asesor; convierte las filas de los CSV en documentos.
- `src/tsd/tag.py`: fase TAG; etiquetado de las fuentes con un modelo de lenguaje y taxonomía cerrada.
- `src/tsd/scoring.py`: fase SCORING; cálculo de la relevancia, la centralidad, la redundancia y la autoridad.
- `src/tsd/dedup.py`: fase DEDUP; deduplicación de tipo greedy por similitud coseno usando FAISS.
- `data/`: corpus del proyecto: ocho archivos PDF, siete archivos CSV y un archivo de texto.
- `queries/`: preguntas de evaluación (18 en total).
- `scripts/`: validación del chunking y de los embeddings, benchmark del bloque TSD, auditoría de los pares descartados y evaluación del corpus.
- `tests/`: pruebas de pytest sin conexión (carga, limpieza, chunking, embeddings, CSV, TSD, informe y corpus de preguntas).
- `docs/`: documentación técnica (incluye el anexo de funciones).
- `output/`: índice de ChromaDB, archivo `embeddings.json` e informes generados (directorio excluido del control de versiones).

Para conocer las funciones de cada archivo (firmas y comportamiento), consultar [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md).

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/dcruzpuri/project-rag-hmd-deporte-municipal.git
cd project-rag-hmd-deporte-municipal
python3 -m venv .venv
```

### 2. Crear el entorno virtual e instalar las dependencias

En Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS o Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

<<<<<<< HEAD
### 3. Configurar el archivo `.env`
=======
## Paso 1 — Ejecutar la app

```bash
streamlit run app.py
```

### Funcionalidades
- Chat con historial de conversación
- Contexto y chunks recuperados visibles
- Métricas por consulta (TOP_K, nº chunks, tiempo, modelo)
- Selector de TOP_K en el sidebar
### 3. Configura `.env`
>>>>>>> origin/develop

```bash
cp .env.example .env
```

El proveedor se elige en cada interruptor (`EMBED_PROVIDER`, `TAG_PROVIDER` y `GEN_PROVIDER`; el valor puede ser `ollama`, `huggingface` o `google`):

- **Ollama**: es preciso arrancar Ollama con el modelo de embeddings y el modelo de lenguaje descargados (la dirección base se configura en `OLLAMA_BASE_URL`, cuyo valor por defecto es `http://127.0.0.1:11434`).
- **HuggingFace**: configurar `HF_DEVICE` (el valor `cpu` o `cuda`) y, únicamente si el modelo es de acceso restringido, introducir el token propio en `HF_TOKEN`.
- **Google**: introducir la clave propia en `GOOGLE_API_KEY`.

Nota: si se cambia el modelo de embeddings o el troceado, es preciso **regenerar el índice** con el flag `--recreate-index` (ver la sección [Uso](#uso)).

## Uso

### Interfaz web

```bash
python -m src.pipeline data --recreate-index
```

La consola muestra el avance fase a fase; las líneas sin marca de tiempo son el detalle que emiten los módulos internos (por ejemplo, una línea de etiquetado por fuente o una línea de inserción por lote):

```text
2026-09-01 12:00:00 [PREFLIGHT] modelo sentence-transformers/all-MiniLM-L6-v2 disponible en ollama
2026-09-01 12:00:00 [LOAD] 234 documentos cargados
2026-09-01 12:00:01 [CLEAN] 234 documentos normalizados
2026-09-01 12:00:02 [TAG] 16 fuentes etiquetadas
2026-09-01 12:00:03 [CHUNK] 16214 chunks | min 40 · p25 612 · media 745 · p75 998 · max 1000 · cortos (menores de 50): 12
2026-09-01 12:00:05 [EMBED] 16214 vectores de 384 dims (EMBED_DIM declarado: 384)
2026-09-01 12:00:06 [TSD] scoring + dedup
[DEDUP] umbral 0.93, descarta 1437 de 16214 (8.9%); passthrough (política exacta) 14040, descartados exactos 0, semánticos 2174
2026-09-01 12:00:07 [INDEX] insertado 14777/14777 (14777 vectores, 10.2 s) (Restante: 0m 0s)
[INDEX] colección 'deporte_municipal': 14777 vectores (14777 en esta inserción)
2026-09-01 12:00:07 [FIN] pipeline completado en 261.05 s
2026-09-01 12:00:07 [INFORME] informe generado: output/informe_index_0a1b2c3d_20260901_1200.md
```

Además, la función `ejecutar_pipeline` devuelve un diccionario de métricas con las claves `num_documentos_cargados`, `num_chunks_pre_dedup` y `num_chunks_post_dedup`, `chunks_descartados`, `dim_embedding`, `dim_declarado` y `dim_modelo`, `chunk_stats`, `fuentes`, `integridad`, `scoring` (incluye la cobertura por categoría y los tres tags principales), `dedup`, `indice`, `preflight`, `fases` y `tiempo_total_s`; ese diccionario alimenta directamente el informe y la evaluación.

### Parámetros opcionales del pipeline

La línea de comandos se invoca de la siguiente forma:

```text
python -m src.pipeline [-h] [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP]
    [--persist-dir PERSIST_DIR] [--collection COLLECTION] [--recreate-index] rutas [rutas ...]
```

Los parámetros disponibles son los siguientes:

- `rutas`: archivos o carpetas a indexar (por ejemplo, `data`); no tiene valor por defecto y es obligatorio indicarlo.
- `--chunk-size`: longitud del chunk; por defecto toma el valor de `CHUNK_SIZE` definido en el archivo `.env`.
- `--chunk-overlap`: solapamiento entre chunks; por defecto toma el valor de `CHUNK_OVERLAP` definido en el archivo `.env`.
- `--persist-dir`: directorio donde reside ChromaDB; por defecto toma el valor de `CHROMA_DIR` definido en el archivo `.env`.
- `--collection`: nombre de la colección; por defecto toma el valor de `COLLECTION_NAME` definido en el archivo `.env`.
- `--recreate-index`: borra la colección antes de indexar; por defecto, no se borra.

### Informe de indexación (archivo de salida)

Cada ejecución escribe su informe (salvo cuando se desactiva con `informe=False` en las pruebas): una lectura única que responde a **qué parámetros determinaron esta ejecución** y a **cómo quedaron las métricas**, en siete secciones (parámetros, preverificación, métricas por fase, troceado, índice final, bloque TSD y señales con criterios de decisión). Cada fila de la tabla de parámetros corresponde a un parámetro que de verdad influyó en la ejecución; las credenciales no se vuelcan nunca.

Para la descripción ampliada (reglas de la tabla de parámetros, interpretación sección a sección y relación de señales con la condición y la acción recomendada), consultar [`docs/ANEXO_OUTPUTS.md`](docs/ANEXO_OUTPUTS.md).

### Validación (scripts y pruebas)

```bash
# Coherencia de los cortes: similitud coseno entre chunks adyacentes frente a chunks aleatorios
python -m scripts.eval_coherencia_chunks

# Margen de un modelo de embeddings (Ollama arrancado)
python -m scripts.test_embeddings

# Coste real del bloque TSD (requiere output/embeddings.json con EXPORT_EMBEDDINGS=true)
python -m scripts.benchmark_tsd

# Auditoría de pares descartados por la deduplicación (muestreo del archivo JSONL)
python scripts/auditar_dedup.py --fuente 211549-0-juegos-deportivos-actual.txt --top 20

# Pruebas offline de carga, limpieza, chunking, embeddings, CSV, TSD e informe
python -m pytest tests/ -v
```

Además, el archivo `scripts/evaluar_rag_corpus.py` calcula métricas de embeddings, chunks y recuperación a partir del volcado del pipeline, y `scripts/generar_eval_rag_en_indexado.py` genera un conjunto de referencia (preguntas y fuentes relevantes) con un modelo de lenguaje durante el indexado.

## Configuración (archivo `.env`) sin tocar los archivos de código

Todo lo configurable reside en `config.py`, que lee el archivo `.env` con valores por defecto (los de la plantilla son los de `.env.example`). Para empezar basta con editar estas variables en el archivo `.env`:

- `EMBED_PROVIDER` (valor por defecto, `ollama`): proveedor de embeddings, entre `ollama`, `huggingface` y `google`. Las variables `TAG_PROVIDER` y `GEN_PROVIDER` funcionan de la misma forma; el etiquetado hereda el valor de generación cuando está vacío.
- Las variables de modelo por proveedor e interruptor (`<PROVEEDOR>_EMBED_MODEL`, `<PROVEEDOR>_TAG_MODEL` y `<PROVEEDOR>_GEN_MODEL`, con sus valores por defecto en `.env.example`), donde el nombre del proveedor es `OLLAMA`, `HF` o `GOOGLE`: modelo de cada interruptor según el proveedor (Ollama, HuggingFace o Google).
- `OLLAMA_BASE_URL` (valor por defecto, `http://127.0.0.1:11434`): solo en caso de usar el proveedor Ollama.
- `HF_DEVICE` (valor por defecto, `cpu`): solo en caso de usar HuggingFace; admite los valores `cpu` y `cuda` (este último exige una tarjeta NVIDIA). La variable `HF_TOKEN` solo es necesaria para los modelos restringidos.
- `GOOGLE_API_KEY` (valor por defecto, vacía): solo en caso de usar Google.
- `CHUNK_SIZE` y `CHUNK_OVERLAP` (valores por defecto, `1000` y `150`): parámetros del troceado.
- `EMBED_DIM` (valor por defecto, `384`): tope de la dimensión del índice; la dimensión final es el mínimo entre la dimensión del modelo y esta constante.
- `TAG_SCORING_DEDUP` (valor por defecto, `true`): activa o desactiva el bloque TSD completo.
- `DEDUP_UMBRAL` (valor por defecto, `0.93`): umbral de similitud coseno para la deduplicación semántica.
- `DEDUP_AUDIT` (valor por defecto, `true`) y `DEDUP_AUDIT_RUTA` (valor por defecto, `output/dedup_audit.jsonl`): activan y enmarcan la auditoría de pares descartados.

Para el resto de variables (la constante `EMBED_BATCH_SIZE`, la memoria caché offline de la preverificación con las constantes `EMBED_DIM_MAX_*`, la constante `EXPORT_EMBEDDINGS`, las constantes `CHROMA_DIR` y `COLLECTION_NAME`), la referencia completa está en [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md) para los valores que resuelve `config.py` y en [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md) para el efecto que tiene cada constante; la plantilla canónica, con cada valor por defecto comentado, es el archivo [`.env.example`](.env.example).

Nota importante: si se cambian las constantes `EMBED_*` o `CHUNK_*`, o el propio corpus, hay que volver a ejecutar el pipeline con el flag `--recreate-index`. Los vectores son función directa del texto troceado; un índice no regenerado produce resultados inválidos.

## Cómo funciona cada fase

- **Preverificación** (módulo `src/embed.py`): comprueba que el modelo indicado por `EMBED_MODEL` existe en el proveedor indicado; se interrumpe si no existe, y degrada a la memoria caché `EMBED_DIM_MAX_*` en el caso de que no pueda verificarse en línea.
- **Carga** (módulo `src/load.py`): carga archivos PDF, TXT, MD o CSV (archivo único o carpeta), con detección automática de la codificación, y siempre puebla los metadatos `source` y `file_hash` (resumen SHA-256 del archivo).
- **Capa CSV** (módulos `csv_advisor.py` y `csv_transform.py`): clasifica cada archivo CSV y lo convierte en documentos dotados de su política de deduplicación; la deduplicación de los archivos CSV nunca es semántica.
- **Limpieza** (módulo `src/clean.py`): normaliza saltos de línea, espacios, carácter de orden al inicio (BOM) y caracteres de control; no toca los metadatos.
- **Etiquetado** (módulo `src/tsd/tag.py`): el modelo de lenguaje etiqueta **una sola vez por fuente** mediante una taxonomía cerrada y propaga las categorías `doc_category`, los tags y la relevancia, además de los valores booleanos de cada tag (contrato de filtrado de la fase online).
- **Troceado** (módulo `src/chunk.py`): usa el divisor recursivo de texto por caracteres, dividiendo primero en párrafos y luego en frases y letras a falta de otro límite; hereda los metadatos del documento padre y respeta la pista de troceado `chunking_hint` de la capa CSV (las entidades `no_chunk` y los grupos ligeros se indexan íntegros, sin trocear).
- **Vectorización** (módulo `src/embed.py`): genera los vectores por proveedor, normalizados y con la dimensión garantizada (ver la sección de [características](#características-principales)).
- **Puntuación** (módulo `src/tsd/scoring.py`): asigna a cada chunk su puntuación semántica mediante una búsqueda de los dos vecinos más cercanos con FAISS.
- **Deduplicación** (módulo `src/tsd/dedup.py`): descarta chunks según la puntuación (método greedy); los archivos con claves exactas (los CSV) solo se deduplican por clave.
- **Indexación** (módulo `src/index.py`): inserta en ChromaDB por lotes, sanea los metadatos y verifica que los identificadores estén vivos en la colección.
- **Informe** (módulo `src/informe.py`): escribe el archivo del informe de indexación con los parámetros, las métricas y las señales.

Cada fase se apoya en los **metadatos** que dejó la fase anterior: el etiquetado los escribe, el troceado los propaga a los fragmentos, la puntuación los lee y los enriquece con la puntuación semántica, la deduplicación los consulta para ordenar el descarte y la indexación los sanea antes de persistirlos en ChromaDB. Para el detalle por funciones (firmas y comportamiento), consultar [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md); para el estado completo del pipeline (flujo de invocación y métricas por consola), la referencia [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md).

### Criterios de puntuación semántica (semantic_score)

Cada chunk recibe una puntuación semántica (un valor entre cero y uno) que combina cuatro señales aditivas y se guarda en el metadato `semantic_score`. La puntuación se calcula como la suma ponderada de: un factor de 0.45 aplicado a la relevancia asignada por el LLM, un factor de 0.20 aplicado a la centralidad, un factor de 0.20 aplicado a la no redundancia y un factor de 0.15 aplicado a la autoridad, recortada al intervalo de cero a uno.

- **La relevancia asignada por el LLM** (peso 0.45): es la valoración de relevancia, entre cero y uno, que el LLM asignó a la fuente en la fase de etiquetado (¿responde a preguntas de tarifas, reservas, horarios y normas?). Tiene un peso superior al de las demás señales porque es la única alineada directamente con el objetivo de RAG, y no solo con la geometría de los embeddings.
- **La centralidad** (peso 0.20): es el coseno del chunk contra el centroide de la colección (la media de todos los vectores), que recompensa el contenido representativo del corpus y penaliza lo marginal.
- **La no redundancia** (peso 0.20): mide cuánto se diferencia el chunk de su vecino más parecido (búsqueda con FAISS de dos vecinos, excluyendo el propio); si el contenido ya existe en otro chunk, no aporta señal nueva.
- **La autoridad** (peso 0.15): es un ajuste fino según el tipo de fuente (reglamento o normativa, valor 1.0; precios o tarifas, valor 0.9; agenda, valor 0.6; el resto, valor 0.7). Una norma pesa más que un archivo CSV genérico, pero no puede compensar una baja relevancia.

La redundancia se calcula con FAISS (búsqueda de dos vecinos) en vez de materializar la matriz completa de similitud, de manera que la memoria crece linealmente con el número de chunks y su dimensión, y no de forma cuadrática (aproximadamente 1.3 GB frente a 68 GB en la escala real). La puntuación alimenta la deduplicación greedy de la fase DEDUP y, en el mapa de ruta, la reordenación de la recuperación. Para conocer el porqué del peso de cada parte y cómo se mide, consultar la sección de *scoring semántico* en [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md); para la implementación, [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md).

### Clasificación de los archivos CSV (csv_advisor)

Los archivos CSV no se tratan igual que el resto: antes de trocear, el módulo `src/csv_advisor.py` decide el tratamiento de cada uno (la etiqueta `[CSV]` se muestra por consola). Son dos niveles de decisión, de lo más económico a lo más costoso: en primer lugar, la **receta de la fuente conocida** (la constante `RECETAS_CONOCIDAS`, con una confianza de 0.99); y, si el esquema cambió o la fuente es nueva, la **heurística por perfil estructural** (cardinalidades, densidad numérica y columnas de identificador, medida, temporal o narrativa). Un archivo CSV desconocido queda en la política conservadora `unknown_csv` o `exact_only` (confianza 0.4): no rompe nada y queda registrado para su revisión. La deduplicación de los archivos CSV **nunca es semántica**; se realiza por clave exacta. La tabla de decisiones del corpus actual, la distinción entre tablas de entidad y tablas de hechos y el glosario de los registros están en [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md).

## Fase online — Recuperación, generación y CLI

La fase online convierte una pregunta del usuario en una respuesta anclada al corpus, con fuentes citadas y abstención cuando no hay evidencia. Está implementada por los módulos siguientes:

- `src/retrieve.py`: recuperación semántica desde ChromaDB; convierte la pregunta en un embedding (con el mismo modelo que el índice) y devuelve los `top-k` chunks más similares con su distancia y su fuente.
- `src/prompts.py`: construcción del prompt final con bloques claramente delimitados (`=== INSTRUCCIONES ===`, `=== CONTEXTO ===`, `=== PREGUNTA ===`, `=== RESPUESTA ===`) y grounding estricto; el mensaje literal de abstención es la constante `ABSTENTION_MESSAGE`.
- `src/generate.py`: capa de generación con el SDK de Google (`google-genai`) usando el modelo `gemini-3.6-flash`, con reintentos automáticos ante errores transitorios (`503`, `429`, `UNAVAILABLE`, `RESOURCE_EXHAUSTED`).
- `src/logging_utils.py`: registro estructurado por consulta; cada pregunta emite una línea JSON con la pregunta, el valor de `top_k`, el número de chunks que entraron al prompt, el modelo, si hubo abstención y los tiempos por fase.
- `src/logic.py`: orquestador de la fase online; expone las funciones `responder()` y `rag_ask()` que se describen más adelante.
- `main.py`: interfaz de línea de comandos con los subcomandos `--query`, `--ask`, `--index` y `--prepare`.

### Uso desde la línea de comandos

```bash
# Recuperación pura (sin LLM): top-k chunks con distancia y fuente
python main.py --query "¿Qué piscinas municipales hay en Madrid?" --top-k 3

# RAG completo: respuesta con fuentes y métricas
python main.py --ask "¿Qué piscinas municipales hay en Madrid?" --top-k 5

# RAG completo con salida JSON (para automatizar evaluación o integrarlo con scripts)
python main.py --ask "¿Cuál es la capital de Francia?" --json

# Indexación del corpus
python main.py --index --recreate-index

El comportamiento del comando --ask se puede resumir así:

Elemento	Descripción
Respuesta	Texto generado por Gemini usando exclusivamente el contenido del corpus
Fuentes	Lista única de archivos fuente usados en el prompt
Métricas	top_k, n_chunks, model, retrieval (segundos), generation (segundos)
Abstención	Booleano; true si el sistema se abstuvo
Error	Mensaje si algo falla antes de llamar al LLM
API interna
Además de la línea de comandos, la fase online expone dos funciones reutilizables sin UI. La interfaz de Streamlit (David) las consume directamente:

from src.logic import responder, rag_ask

resultado = responder("¿Qué piscinas municipales hay en Madrid?", top_k=5)
# resultado["respuesta"]  -> str
# resultado["fuentes"]    -> list[str]
# resultado["contexto"]   -> str (texto de los chunks para depuración)
# resultado["chunks"]     -> list[dict]
# resultado["metrics"]    -> dict (top_k, n_chunks, model, retrieval, generation)
# resultado["abstained"]  -> bool
# resultado["error"]      -> str | None

texto = rag_ask("¿Qué descuentos hay para abonados?")
# -> solo la respuesta en texto (útil para el módulo de Agentes)

Grounding y abstención
El sistema implementa dos capas de abstención independientes:

Guardrail de distancia (src/logic.py): si la mejor distancia coseno del top-k supera el umbral UMBRAL_ABSTENCION = 0.65, el sistema se abstiene sin llamar al LLM. Esto ahorra tokens y latencia cuando la pregunta está fuera del dominio del corpus.

Prompt restrictivo (src/prompts.py): si el LLM recibe contexto pero no encuentra información suficiente, devuelve literalmente el valor de ABSTENTION_MESSAGE.

Además, el prompt obliga a citar las fuentes cuando el contexto las incluye.

Comportamiento verificado
Pregunta	Resultado
"¿Qué piscinas municipales hay en Madrid?"	✅ Responde con 18 piscinas y fuentes citadas
"¿Qué instalaciones deportivas hay en Chamberí?"	⚠️ Abstención (los chunks no mencionan el distrito)
"¿Qué descuentos hay para abonados?"	⚠️ Abstención (el CSV es de hechos, no documenta)
"¿Cuál es la capital de Francia?"	✅ Abstención sin llamar al LLM (guardrail de distancia)
Configuración específica de la fase online
Las variables del archivo .env que controlan la fase online son las siguientes:

GEN_PROVIDER (valor por defecto, google): proveedor del LLM de generación; los valores admitidos son google, huggingface y ollama.

GOOGLE_GEN_MODEL (valor por defecto, gemini-3.6-flash): nombre del modelo de generación.

GOOGLE_API_KEY: clave de Google AI Studio (obligatoria para la generación con Google).

TOP_K (valor por defecto, 5): número de chunks recuperados por consulta.

MAX_CHUNKS (valor por defecto, 5): número máximo de chunks que entran al prompt.

GEN_TEMPERATURE (valor por defecto, 0.2): temperatura del LLM; se mantiene baja para reforzar el grounding.

ABSTENTION_MESSAGE: mensaje literal que se devuelve cuando no hay evidencia suficiente.

Registro por consulta
Cada consulta emite por consola una línea con formato JSON que resume la ejecución. Un ejemplo:

{
  "pregunta": "¿Qué piscinas municipales hay en Madrid?",
  "top_k": 5,
  "n_chunks": 5,
  "modelo": "google:gemini-3.6-flash",
  "abstained": false,
  "t_retrieval": 11.1,
  "t_generation": 8.1
}

Este registro alimenta el informe de evaluación y la tabla de métricas de la interfaz de Streamlit.

Dependencias adicionales
Además de las dependencias de requirements.txt, la fase online requiere:

# Necesario para la deduplicación (bloque TSD)
pip install faiss-cpu

# Necesario para usar GPU NVIDIA durante la indexación
pip install torch --index-url https://download.pytorch.org/whl/cu121

Sin torch con CUDA, la indexación cae a CPU y su tiempo se multiplica por cinco o diez.

Nota sobre el modelo de generación: el modelo gemini-2.0-flash fue retirado por Google; la constante GOOGLE_GEN_MODEL se ha actualizado a gemini-3.6-flash.
## Corpus y fuentes

Corpus de la sede de **datos abiertos del Ayuntamiento de Madrid** (descargado en septiembre de 2026). Datos públicos y de uso educativo:

- El archivo `PreciosPublicos2026.pdf` procede de [Precios públicos centros deportivos 2026](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/ficheros/TarifasD/PreciosPublicos2026.pdf).
- El archivo `Tarifas_deportivas.pdf` procede de [Tarifas de servicios en centros deportivos](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/ficheros/TarifasD/Tarifas_deportivas.pdf).
- El archivo `reglamento_instalaciones.pdf` es el [Reglamento de instalaciones deportivas](https://sede.madrid.es/eli/es-md-01860896/reg/2012/10/15/(1)/dof/spa/pdf).
- El archivo `PiscinasAireLibre2026.pdf` son las [Piscinas de verano 2026 aire libre](https://www.madrid.es/UnidadesDescentralizadas/Deportes/EspecialInformativo/Verano2026/ficheros/PiscinasAireLibre2026.pdf).
- El archivo `DecretoAnulacionReservasConCoste.pdf` es el [Decreto de anulación de reservas con coste](https://www.madrid.es/UnidadesDescentralizadas/Deportes/ContenidoGenerico/ContenidoGenerico2024/Ficheros/DecretoAnulacionReservasConCoste.pdf).
- Los archivos `normativaGeneral46jdm.pdf` y `BasesDeportesEquipo47jdm_1Sp.pdf` son la [Normativa de los Juegos Deportivos Municipales](https://www.madrid.es/UnidadesDescentralizadas/Deportes/EspecialInformativo/46%20Juegos%20Deportivos%20Municipales%252025-26/normativas/normativaGeneral46jdm.pdf).
- El archivo `20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf` es la [infografía del Abono Deporte Madrid](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Faq/ficheros/infografias%20faq%202025/20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf).
- El archivo `200186-0-polideportivos.csv` corresponde al conjunto de datos [Polideportivos](https://datos.madrid.es/dataset/200186-0-polideportivos).
- El archivo `200215-0-instalaciones-deportivas.csv` corresponde al conjunto de datos [Instalaciones deportivas básicas](https://datos.madrid.es/dataset/200215-0-instalaciones-deportivas).
- El archivo `210227-0-piscinas-publicas.csv` corresponde al conjunto de datos [Piscinas públicas](https://datos.madrid.es/dataset/210227-0-piscinas-publicas).
- El archivo `300390-0-areas-deportivas.csv` corresponde al conjunto de datos [Áreas de actividades deportivas](https://datos.madrid.es/dataset/300390-0-areas-deportivas).
- El archivo `212504-0-agenda-actividades-deportes.csv` corresponde al conjunto de datos [Agenda de actividades deportivas](https://datos.madrid.es/dataset/212504-0-agenda-actividades-deportes).
- El archivo `300085-0-deportes_abonos.csv` corresponde al conjunto de datos [Abonados en centros deportivos](https://datos.madrid.es/dataset/300085-0-deportes_abonos).
- El archivo `300097-0-deportes-descuentos.csv` corresponde al conjunto de datos [Descuentos en instalaciones](https://datos.madrid.es/dataset/300097-0-deportes-descuentos).
- El archivo `211549-0-juegos-deportivos-actual.txt` corresponde al conjunto de datos [Juegos deportivos municipales vigente](https://datos.madrid.es/dataset/211549-0-juegos-deportivos-actual).

Los archivos CSV pueden regenerarse descargando cualquiera de los conjuntos de datos del [grupo deporte de datos.madrid.es](https://datos.madrid.es/group/deporte); el cargador los recarga igual aunque cambien de esquema, siempre que se identifiquen adecuadamente.

## Documentación técnica

- [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md): **funciones de cada archivo** (módulos fuente, bloque TSD, capa de CSV, scripts y pruebas), con firmas y comportamiento clave.
- [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md): estado completo del pipeline: módulos, flujo de invocación, proveedores y métricas por consola.
- [`docs/ANEXO_CHUNKING.md`](docs/ANEXO_CHUNKING.md): cómo cambiar el nivel de troceado y la validación obligatoria (incluye errores comunes).
- [`docs/ANEXO_OUTPUTS.md`](docs/ANEXO_OUTPUTS.md): salidas del pipeline, esto es, el informe de indexación (secciones, reglas de parámetros y señales con acción recomendada) y el archivo `output/embeddings.json`.
- [`docs/HIGHLIGHTS_CORPUS_DATA.md`](docs/HIGHLIGHTS_CORPUS_DATA.md): los apartados superados frente al enunciado y la guía de partida (preverificación, bloque TSD, dimensión garantizada e informe).

## Roadmap

Completado en la versión actual:

- Corpus multi formato (PDF, CSV y TXT).
- Pipeline offline completo con bloque TSD.
- Proveedor múltiple (Ollama, HuggingFace y Google).
- Índice en ChromaDB verificado y regenerable.

Pendiente (fase online):

- Recuperación: el flag `--query` con filtrado por `doc_category` y por los booleanos de cada tag (ya presentes en el índice mediante el filtrado `where` de Chroma) y reordenación por la puntuación semántica.
- Generación con RAG y abstención: el flag `--ask`.
- Interfaz de Streamlit: chat y chunks visibles, y tabla de métricas.

## Solución de problemas

- **El pipeline se interrumpe en la preverificación con un mensaje PREFLIGHT que indica que el modelo no está disponible en el proveedor seleccionado.** Causa probable: el modelo no existe en el proveedor seleccionado (error tipográfico, o Ollama no cuenta con él descargado). Solución: arrancar el servicio con el modelo descargado (mediante el comando `ollama pull` seguido del nombre del modelo) o corregir la constante `EMBED_*_MODEL` correspondiente en el archivo `.env`.
- **El pipeline emite un aviso PREFLIGHT de que la verificación online no pudo completarse (red cortada o ausencia de clave API).** El modelo se asume disponible a través de la memoria caché `EMBED_DIM_MAX_*`. Es un mensaje informativo y el proceso continúa; para blindarlo por completo, conviene declarar la constante `EMBED_DIM_MAX_` del proveedor en el archivo `.env`, arrancar el servicio y configurar la clave `GOOGLE_API_KEY` si el proveedor es Google.
- **El pipeline emite un aviso AVISO de que `EMBED_DIM` es mayor que la dimensión generada.** Causa probable: el valor de `EMBED_DIM` es mayor que la dimensión que genera el modelo. El índice se creó con la dimensión generada, y el valor declarado queda mostrado por separado en la tabla de parámetros del informe (la fila `EMBED_DIM_DECLARADO`). Solución: fijar `EMBED_DIM` en la dimensión del modelo (por ejemplo, `384` para el modelo `all-MiniLM-L6-v2`) en el archivo `.env`.
- **El pipeline emite un mensaje INFO de que `EMBED_DIM` es menor que la dimensión máxima del modelo.** Causa probable: el valor de `EMBED_DIM` es menor que la dimensión máxima declarada. Es un mensaje informativo: el índice se creó con la dimensión declarada y el modelo puede manejar hasta la dimensión máxima; para indexar con más dimensiones, conviene aumentar `EMBED_DIM` y regenerar.
- **Ollama responde sin el campo de embeddings.** Causa probable: versión antigua del servicio. Solución: actualizar Ollama a al menos la versión 0.9 (el punto de servicio por lotes `/api/embed` lo exige) o cambiar a `EMBED_PROVIDER=huggingface`.
- **ChromaDB emite el error de tamaño de lote máximo superado (el mensaje indica `greater than max batch size`).** Causa probable: colección dañada (el código ya inserta por lotes). Solución: borrar el directorio `output/` y volver a generar el índice.
- **La fase de etiquetado usa la categoría por defecto `instalaciones`.** Causa probable: el LLM devolvió un objeto JSON malformado. Solución: revisar que el modelo de `TAG_PROVIDER` sea capaz de emitir JSON válido. Para el pipeline no es un error: se marca en la consola y el proceso continúa.
- **La deduplicación descarta casi todo el corpus** (por ejemplo, más del noventa por ciento con `DEDUP_UMBRAL` fijado en 0.93). Causa probable: el umbral es ajustable y, en los archivos CSV con chunking por fuente, las fuentes repetitivas se deduplican a gran escala. Solución: bajar el umbral (por ejemplo, `0.85`) para conservar más chunks, o subirlo (por ejemplo, `0.95`) para deduplicar más. Para inspeccionar los pares descartados, consultar la auditoría `output/dedup_audit.jsonl` con `scripts/auditar_dedup.py`.
- **Se cambió el troceado o el modelo y la recuperación falla.** Causa probable: el índice no fue regenerado. Solución: ejecutar `python -m src.pipeline data --recreate-index`.
