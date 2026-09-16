# project-rag-hmd-deporte-municipal

La interfaz web permite interactuar con el sistema RAG de forma visual.

## Paso 0 — Instalación y arranque
# Retrieval Augmented Generation (RAG) para Deporte Municipal del Ayuntamiento de Madrid

Sistema **RAG (Retrieval Augmented Generation)** para responder preguntas basadas en recuperación de contenido sobre documentos de **deporte municipal de Madrid**: tarifas, abonos, reservas, horarios e instalaciones.

Carga un corpus real de documentos (PDF + CSV + TXT) de la sede de datos abiertos del Ayuntamiento de Madrid, lo trocea, genera embeddings, evalúa y mejora la calidad semántica con etiquetado LLM + scoring + deduplicación (bloque **TSD** que se explica más adelante, opcional) y lo guarda en la base de datos vectorial **ChromaDB** para retrieval semántico. Pipeline **offline** completo; la fase online (retrieval + generación + UI) está en el [roadmap](#roadmap).



```text
PREFLIGHT → LOAD → CLEAN → TAG (LLM) → CHUNK → EMBED → SCORING → DEDUP → INDEX → INFORME
```

## Características principales

- **Preflight de disponibilidad**: el pipeline comprueba que `EMBED_MODEL` está disponible en `EMBED_PROVIDER` (Ollama `/api/tags` · HuggingFace `repo_exists` · Google listado de modelos) y **aborta** si no está. Si no se puede verificar online (red cortada / sin API key) degrada a caché offline (`EMBED_DIM_MAX_*`) con aviso, en vez de romper.
- **Multi-proveedor** sin tocar código: `ollama` / `huggingface` / `google` por variable de entorno, con **tres interruptores independientes** por proveedor (EMBED_provider, TAG_provider, GEN_provider; TAG hereda de GEN por defecto si no se define y donde `provider` se corresponde con uno de los tres anteriores).
- **Bloque TSD** (Tag–Scoring–Dedup, opcional): primero etiqueta cada fuente con un LLM mediante una taxonomía cerrada. Después asigna a cada chunk un `semantic_score` que combina relevancia, centralidad, redundancia y autoridad (calculada con FAISS). Por último, aplica una deduplicación *greedy*: ordena los chunks por puntuación y conserva cada uno únicamente cuando no es demasiado parecido a los que ya se han conservado. Este proceso respeta la política específica de los CSV, de modo que el índice contiene más información útil y menos ruido.
- **Tratamiento diferencial de CSVs**: `csv_advisor` analiza y clasifica cada CSV (entidad con ID estable / tabla de hechos / textual / desconocido) y decide cómo convertirlo en documento para indexación y decidir cómo aplicar la deduplicación; la deduplicación de CSVs **nunca es semántica** (se realiza por clave exacta). [Detalle](docs/ANEXO_FUNC.md#3-capa-csv-clasificación-diferencial).
- **Dimensión garantizada**: la dimensión final de todos los vectores del índice es siempre `min(dim del modelo, EMBED_DIM)`. Si el proveedor lo permite, se solicita esa dimensión en el servidor (Ollama mediante `dimensions` y HuggingFace mediante `truncate_dim`); después, el cliente recorta el prefijo del vector si es necesario y lo renormaliza. Si `EMBED_DIM` es menor que la dimensión del modelo se informa con `INFO`; si es mayor, se muestra un `AVISO` y se usa la dimensión real del modelo, **evitando incompatibilidades en ChromaDB**.
- **Índice regenerable** en ChromaDB persistente (métrica coseno): cada ejecución puede reconstruir el índice a partir del corpus, sin depender de inserciones anteriores. Los vectores se insertan por lotes para controlar el uso de memoria y mejorar el rendimiento. Al finalizar, el pipeline vuelve a consultar los identificadores insertados y compara el resultado con lo esperado; así detecta inserciones incompletas o reindexaciones parciales antes de dar el proceso por válido.
- **Informe de indexación** en markdown al terminar cada ejecución (`output/informe_index_<GUID8>_YYYYMMDD_HHMM.md`, donde `GUID8` son los ocho primeros caracteres del GUID de ChromaDB): parámetros aplicados, preflight, tiempos por fase, chunking, índice, TSD y **señales automáticas** con la acción recomendada.
- **Logging por consola** de cada fase y `dict` de métricas devuelto por `ejecutar_pipeline` listo para evaluación.

Preguntas de ejemplo (el corpus debe responder a las 18 de [`queries/preguntas.json`](queries/preguntas.json)):

- ¿Cuánto cuesta el abono de piscina?
- ¿Hay descuentos para menores, mayores, estudiantes o personas en paro?
- ¿Puedo reservar siendo no empadronado?
- ¿Qué pasa si cancelo una reserva con coste?

Entre otras.

## Estructura del proyecto

```text
project-rag-hmd-deporte-municipal/
├── README.md
├── requirements.txt
├── .env.example           # copia a .env y edita (defaults de cada variable)
├── config.py              # todas las variables de .env resueltas a constantes
├── src/
│   ├── pipeline.py        # orquestador: encadena todas las fases + `dict` de métricas
│   ├── load.py            # LOAD: PDF/TXT/MD/CSV → Documents (archivo o carpeta)
│   ├── clean.py           # CLEAN: normaliza saltos, espacios, BOM, control chars
│   ├── chunk.py           # CHUNK: RecursiveCharacterTextSplitter (Level 2)
│   ├── embed.py           # EMBED: multi-proveedor (ollama / huggingface / google)
│   ├── index.py           # INDEX: ChromaDB persistente, métrica coseno, verificado
│   ├── informe.py         # renderiza output/informe_index_<GUID8>_YYYYMMDD_HHMM.md
│   ├── csv_advisor.py     # clasifica cada CSV y decide tratamiento + dedup_policy
│   ├── csv_transform.py   # materializa el consejo: filas → Documents
│   └── tsd/
│       ├── tag.py         # TAG: etiquetado LLM por fuente (taxonomía cerrada)
│       ├── scoring.py     # SCORING: relevancia + centralidad + redundancia + autoridad
│       └── dedup.py       # DEDUP: greedy por similitud coseno (FAISS)
├── data/                   # corpus: 8 PDF + 7 CSV + 1 TXT
├── queries/                # preguntas de evaluación (18)
├── scripts/                # validación de chunking, embeddings y benchmark TSD
├── tests/                  # pruebas pytest offline (load/clean/chunk/embed/csv/tsd/informe)
├── docs/                   # documentación técnica (incluye ANEXO_FUNC)
└── output/                 # índice Chroma, embeddings.json, informe (gitignored)
```

Funciones de cada archivo (firmas y comportamiento): [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md).

## Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/dcruzpuri/project-rag-hmd-deporte-municipal.git
cd project-rag-hmd-deporte-municipal
python3 -m venv .venv
```

### 2. Crea el entorno virtual e instala dependencias

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

```bash
cp .env.example .env
```

Elige el proveedor en cada interruptor (`EMBED_PROVIDER` / `TAG_PROVIDER` / `GEN_PROVIDER`: `ollama`, `huggingface` o `google`):

- **`ollama`**: arranca Ollama con el modelo de embeddings y el LLM descargados (`OLLAMA_BASE_URL`, por defecto `http://127.0.0.1:11434`).
- **`huggingface`**: configura `HF_DEVICE` (`cpu` o `cuda`) y, solo si el modelo es gated, pon tu token en `HF_TOKEN`.
- **`google`**: pon tu clave en `GOOGLE_API_KEY`.

> **Ojo:** si cambias el modelo de embeddings o el chunking, **regenera el índice** con `--recreate-index` (ver [Uso](#uso)).

## Uso

### Indexar el corpus

Desde la raíz del proyecto:

```bash
python -m src.pipeline data --recreate-index
```

La consola va mostrando el avance fase a fase:

```text
2025-09-01 12:00:00 [PREFLIGHT] modelo qwen3-embedding-4b disponible en ollama
2025-09-01 12:00:00 [LOAD] 234 documentos cargados
2025-09-01 12:00:01 [CLEAN] 234 documentos normalizados
2025-09-01 12:00:02 [TAG] 16 fuentes etiquetadas
2025-09-01 12:00:03 [CHUNK] 16214 chunks | min 40 · p25 612 · media 745 · p75 998 · max 1000 · cortos(<50) 12
2025-09-01 12:00:05 [EMBED] 16214 vectores de 2560 dims (EMBED_DIM declarado: 2560)
2025-09-01 12:00:06 [DEDUP] umbral 0.93 -> descarta 1437 de 16214 (8.9%) | passthrough (política exacta) 14040 descartados exactos 0 · semánticos 2174
2025-09-01 12:00:07 [INDEX] colección 'deporte_municipal': 14777 vectores (14777 en esta inserción)
2025-09-01 12:00:07 [FIN] pipeline completado en 261.05 s
2026-09-01 12:00:07 [INFORME] informe generado: output\informe_index_0a1b2c3d_20260901_1200.md
```

`ejecutar_pipeline` devuelve además un `dict` de métricas (`num_documentos_cargados`, `num_chunks_pre_dedup`, `num_chunks_post_dedup`, `chunks_descartados`, `dim_embedding`, `chunk_stats`, `scoring`, `dedup` —incluye cobertura por categoría y top-3 tags—, `indice`, `tiempo_total_s`) listo para el informe y para evaluación.

### Parámetros opcionales del pipeline

```text
python -m src.pipeline [-h] [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP]
    [--persist-dir PERSIST_DIR] [--collection COLLECTION] [--recreate-index] rutas [rutas ...]
```

| Flag | Qué hace | Default |
|---|---|---|
| `rutas` | Archivos o carpetas a indexar (p. ej. `data`) | — |
| `--chunk-size` | Longitud de chunk | `CHUNK_SIZE` de `.env` |
| `--chunk-overlap` | Sobrelap de chunks | `CHUNK_OVERLAP` de `.env` |
| `--persist-dir` | Directorio de ChromaDB | `CHROMA_DIR` de `.env` |
| `--collection` | Nombre de la colección | `COLLECTION_NAME` de `.env` |
| `--recreate-index` | Borra la colección antes de indexar | no |

### Informe de indexación (`output/informe_index_<GUID8>_YYYYMMDD_HHMM.md`)

Cada ejecución escribe este informe (salvo `informe=False` en los tests): una lectura única que responde **qué parámetros determinaron esta ejecución** y **cómo quedaron las métricas**, en 7 secciones (parámetros · preflight · métricas por fase · chunking · índice final · TSD · señales y criterios de decisión). Cada fila de la tabla de parámetros corresponde a un parámetro que realmente influyó; las credenciales nunca se vuelcan.

Descripción ampliada — reglas de la tabla de parámetros, interpretación sección a sección y tabla de señales (condición → acción recomendada): [`docs/ANEXO_OUTPUTS.md`](docs/ANEXO_OUTPUTS.md).

### Validación (scripts y tests)

```bash
# Coherencia de los cortes: similitud coseno entre chunks adyacentes vs. aleatorios
python -m scripts.eval_coherencia_chunks

# Margen de un modelo de embeddings (Ollama arrancado)
python -m scripts.test_embeddings

# Coste real de TSD (requiere output/embeddings.json con EXPORT_EMBEDDINGS=true)
python -m scripts.benchmark_tsd

# Pruebas offline de load/clean/chunk/embed/csv/tsd/informe
python -m pytest tests/ -v
```

## Configuración (.env) sin tocar archivos de código

Todo lo configurable vive en `config.py`, que lee `.env` con defaults (los de la tabla rápida son los de `.env.example`). Para empezar basta editar estas variables en `.env`:

| Variable | Default (.env.example) | Qué controla |
|---|---|---|
| `EMBED_PROVIDER` | `huggingface` | Proveedor de embeddings (`ollama` / `huggingface` / `google`). `TAG_PROVIDER` y `GEN_PROVIDER` funcionan igual (TAG hereda GEN si está vacío). |
| `<PROVEEDOR>_EMBED_MODEL` / `_TAG_MODEL` / `_GEN_MODEL` | ver `.env.example` | Modelo de cada interruptor por proveedor (Ollama / HuggingFace / Google). |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Solo si usas `ollama`. |
| `HF_DEVICE` | `cuda` | Solo si usas `huggingface`: `cpu` / `cuda` (cuda exige GPU NVIDIA). `HF_TOKEN` solo para modelos gated. |
| `GOOGLE_API_KEY` | (vacía) | Solo si usas `google`. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `100` | Parámetros del troceado. |
| `EMBED_DIM` | `1024` | Cap de dimensión del índice: `dim final = min(dim del modelo, EMBED_DIM)`. |
| `TAG_SCORING_DEDUP` | `false` en `.env.example` | Activa/desactiva el bloque TSD completo (valor de `config.py`: `true`). |
| `DEDUP_UMBRAL` | `0.93` | Umbral coseno de dedup semántica. |

Tabla completa (modelos por proveedor e interruptor, `EMBED_BATCH_SIZE`, caché offline del preflight `EMBED_DIM_MAX_*`, `EXPORT_EMBEDDINGS`, `CHROMA_DIR`, `COLLECTION_NAME`): [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md) §3 (defaults de `config.py`) y [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md) §1 (las constantes y su efecto). El template canónico con defaults comentados: [`.env.example`](.env.example).

**IMPORTANTE:** si cambias `EMBED_*`, `CHUNK_*` o el corpus, ejecuta de nuevo el pipeline con `--recreate-index`. Los vectores son función directa del texto troceado; un índice no regenerado produce resultados inválidos.

## Cómo funciona cada fase

| Fase | Módulo | Qué hace (resumen) |
|---|---|---|
| PREFLIGHT | `src/embed.py` | Verifica que `EMBED_MODEL` existe en el proveedor; aborta si no, degrada a caché `EMBED_DIM_MAX_*` si no se puede comprobar online. |
| LOAD | `src/load.py` | Carga PDF/TXT/MD/CSV (archivo o carpeta), encoding autodetectado, `metadata.source` siempre. |
| CSV | `csv_advisor.py` + `csv_transform.py` | Clasifica cada CSV y lo convierte en documentos con su `dedup_policy` (la dedup de CSVs nunca es semántica). |
| CLEAN | `src/clean.py` | Normaliza saltos/espacios/BOM/control. No toca metadatos. |
| TAG | `tsd/tag.py` | El LLM etiqueta **una vez por fuente** (taxonomía cerrada) y propaga `doc_category`/`tags`/`relevancia_llm` + booleanos `tag_<nombre>` (contrato de filtrado de la fase online). |
| CHUNK | `src/chunk.py` | `RecursiveCharacterTextSplitter` (párrafo→frase→letra); hereda metadata del padre; respeta `chunking_hint`. |
| EMBED | `src/embed.py` | Vectores por proveedor, normalizados, dimensión garantizada (ver [características](#características-principales)). |
| SCORING | `tsd/scoring.py` | `semantic_score` (FAISS `k=2`). |
| DEDUP | `tsd/dedup.py` | Descarte greedy por score; las claves exactas (CSV) solo por clave. |
| INDEX | `src/index.py` | Inserta en ChromaDB por lotes, sanea metadatos y verifica ids vivos. |
| INFORME | `src/informe.py` | Escribe `output/informe_index_<GUID8>_YYYYMMDD_HHMM.md` con parámetros, métricas y señales. |

La dependencia entre fases va por **metadata**: `tag` escribe → `chunk` propaga → `scoring` lee y escribe → `dedup` lee → `index` sanea. Qué hace cada una a nivel de funciones (firmas y comportamiento): [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md) §2. Estado completo del pipeline, flujo de invocación y métricas por consola: [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md).

### Criterios de scoring (`semantic_score`)

`semantic_score = 0.45·relevancia_LLM + 0.20·centralidad + 0.20·(1 − redundancia) + 0.15·autoridad`, señal aditiva en [0, 1] guardada en `metadata['semantic_score']`. Cada componente es interpretable por separado (¿por qué pesa cada parte y cómo se mide: [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md) §2, *scoring semántico*; implementación: [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md) §4). Redundancia por FAISS `k=2` (memoria O(n·d) en vez de O(n²), ~1,3 GB vs ~68 GB en la escala real). Alimenta el **dedup greedy** de `DEDUP` y, en el roadmap, el rerank de retrieval.

### Clasificación de CSVs (`csv_advisor`)

Los CSV no se tratan igual: antes de trocear, `src/csv_advisor.py` decide el tratamiento de cada uno (el log `[CSV]` se muestra por consola). Dos niveles de decisión, de lo barato a lo caro: **receta de fuente conocida** (`RECETAS_CONOCIDAS`, confianza 0.99) y, si el esquema cambió o la fuente es nueva, **heurística por perfil estructural** (cardinalidades, densidad numérica, columnas ID/medida/temporal/narrativa); un CSV desconocido cae al conservador `unknown_csv` / `exact_only` (confianza 0.4): no rompe nada y queda registrado para revisar. La dedup de CSVs **nunca es semántica** (por clave exacta). Tabla de decisiones del corpus actual, tabla de entidad vs. hechos y glosario de los logs: [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md) §3.

## Corpus y fuentes

Corpus de la sede de **datos abiertos del Ayuntamiento de Madrid** (descargado en septiembre de 2026). Datos públicos y/o de uso educativo.

| Archivo | Origen |
|---|---|
| `PreciosPublicos2026.pdf` | [Precios públicos centros deportivos 2026](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/ficheros/TarifasD/PreciosPublicos2026.pdf) |
| `Tarifas_deportivas.pdf` | [Tarifas de servicios en centros deportivos](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/ficheros/TarifasD/Tarifas_deportivas.pdf) |
| `reglamento_instalaciones.pdf` | [Reglamento de instalaciones deportivas](https://sede.madrid.es/eli/es-md-01860896/reg/2012/10/15/(1)/dof/spa/pdf) |
| `PiscinasAireLibre2026.pdf` | [Piscinas de verano 2026 aire libre](https://www.madrid.es/UnidadesDescentralizadas/Deportes/EspecialInformativo/Verano2026/ficheros/PiscinasAireLibre2026.pdf) |
| `DecretoAnulacionReservasConCoste.pdf` | [Decreto anulación de reservas con coste](https://www.madrid.es/UnidadesDescentralizadas/Deportes/ContenidoGenerico/ContenidoGenerico2024/Ficheros/DecretoAnulacionReservasConCoste.pdf) |
| `normativaGeneral46jdm.pdf` / `BasesDeportesEquipo47jdm_1Sp.pdf` | [Normativa Juegos Deportivos Municipales](https://www.madrid.es/UnidadesDescentralizadas/Deportes/EspecialInformativo/46%20Juegos%20Deportivos%20Municipales%2025-26/normativas/normativaGeneral46jdm.pdf) |
| `20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf` | [Infografía Abono Deporte Madrid](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Faq/ficheros/infografias%20faq%202025/20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf) |
| `200186-0-polideportivos.csv` | [Polideportivos](https://datos.madrid.es/dataset/200186-0-polideportivos) |
| `200215-0-instalaciones-deportivas.csv` | [Instalaciones deportivas básicas](https://datos.madrid.es/dataset/200215-0-instalaciones-deportivas) |
| `210227-0-piscinas-publicas.csv` | [Piscinas públicas](https://datos.madrid.es/dataset/210227-0-piscinas-publicas) |
| `300390-0-areas-deportivas.csv` | [Áreas de actividades deportivas](https://datos.madrid.es/dataset/300390-0-areas-deportivas) |
| `212504-0-agenda-actividades-deportes.csv` | [Agenda de actividades deportivas](https://datos.madrid.es/dataset/212504-0-agenda-actividades-deportes) |
| `300085-0-deportes_abonos.csv` | [Abonados en centros deportivos](https://datos.madrid.es/dataset/300085-0-deportes_abonos) |
| `300097-0-deportes-descuentos.csv` | [Descuentos en instalaciones](https://datos.madrid.es/dataset/300097-0-deportes-descuentos) |
| `211549-0-juegos-deportivos-actual.txt` | [Juegos deportivos municipales vigente](https://datos.madrid.es/dataset/211549-0-juegos-deportivos-actual) |

Los CSV pueden regenerarse descargando cualquier dataset de [datos.madrid.es · grupo deporte](https://datos.madrid.es/group/deporte); el loader los recarga igual aunque cambien de esquema siempre que se identifiquen adecuadamente.

## Documentación técnica

| Documento | Contenido |
|---|---|
| [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md) | **Funciones de cada archivo** (src, tsd, capa CSV, scripts, tests) con firmas y comportamiento clave. |
| [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md) | Estado completo del pipeline: módulos, flujo de invocación, proveedores y métricas por consola. |
| [`docs/ANEXO_CHUNKING.md`](docs/ANEXO_CHUNKING.md) | Cómo cambiar el nivel de chunking y la validación obligatoria (incluye errores comunes). |
| [`docs/ANEXO_OUTPUTS.md`](docs/ANEXO_OUTPUTS.md) | Outputs del pipeline: informe de indexación (secciones, reglas de parámetros y señales con acción recomendada) y `output/embeddings.json`. |
| [`docs/HIGHLIGHTS_CORPUS_DATA.md`](docs/HIGHLIGHTS_CORPUS_DATA.md) | Excedentes: qué aporta este bloque por encima de ENUNCIADO/GUIA (pre-flight, TSD, dimensión garantizada, informe). |

## Roadmap

- [x] Corpus multi-formato (PDF + CSV + TXT)
- [x] Pipeline offline completo con bloque TSD
- [x] Multi-proveedor (Ollama / HuggingFace / Google)
- [x] Índice Chroma verificado y regenerable
- [ ] Fase online (retrieval): `--query` con filtro por `doc_category` y booleanos `tag_<nombre>` (ya en el índice: `where` de Chroma) y rerank por `semantic_score`
- [ ] Generación RAG con abstención: `--ask`
- [ ] UI Streamlit: chat + chunks visibles + tabla de métricas

## Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| El pipeline aborta con `PREFLIGHT: el modelo 'X' NO está disponible en <proveedor>` | El modelo no existe en el proveedor seleccionado (typo, o Ollama no lo tiene descargado) | Arranca el servicio con el modelo descargado (p. ej. `ollama pull <modelo>`) o corrige `EMBED_*_MODEL` en `.env`. |
| El pipeline avisa `PREFLIGHT: no se pudo verificar online ...` | Red cortada / sin API key: no se pudo comprobar el modelo (se asume vía caché `EMBED_DIM_MAX_*`) | Informativo: se continúa. Para blindarlo, declara `EMBED_DIM_MAX_<proveedor>` en `.env` y arranca el servicio / pon `GOOGLE_API_KEY`. |
| El pipeline avisa `AVISO: EMBED_DIM (X) supera la dim máxima del modelo (Y)` | `EMBED_DIM` mayor que la dim real del modelo | Pon `EMBED_DIM` a la dim real del modelo (p. ej. `384` para `all-MiniLM-L6-v2`) **y** regenera con `--recreate-index`. El índice se creó con `Y` (el máximo del modelo). |
| El pipeline avisa `INFO: EMBED_DIM (X) es menor que la dim máxima del modelo (Y)` | `EMBED_DIM` menor que la dim máxima declarada | Informativo: el índice se creó con `X` dims; el modelo puede manejar hasta `Y`. Para indexar con más dimensiones, sube `EMBED_DIM` y regenera. |
| Ollama responde sin campo `embeddings` | Versión antigua | Actualiza Ollama a >= 0.9 (el batch `/api/embed` lo exige) o usa `EMBED_PROVIDER=huggingface`. |
| Chroma `greater than max batch size` | Colección dañada (el código ya inserta por lotes) | Borra `output/` y regenera. |
| El TAG usa la categoría por defecto `instalaciones` | El LLM devolvió JSON malformado | Revisa que el modelo de `TAG_PROVIDER` sea capaz de emitir JSON. Para el pipeline no es un error: se marca en la consola y continúa. |
| La dedup descarta casi todo el corpus (p. ej. >90 % con `DEDUP_UMBRAL=0.93`) | El umbral es ajustable | **Bájalo** (p. ej. `0.85`) para conservar más chunks, **súbelo** (p. ej. `0.95`) para dedup más. Con chunking por fuente (CSV), fuentes repetitivas se deduplican a masas. |
| Cambié chunking/modelo y el retrieval falla | Índice no regenerado | Ejecuta `python -m src.pipeline data --recreate-index`. |
