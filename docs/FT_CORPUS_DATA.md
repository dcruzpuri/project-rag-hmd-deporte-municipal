# FT_CORPUS_DATA — Pipeline RAG de deporte municipal

## 1. Qué hace el programa (estado actual)

Pipeline offline RAG que prepara un índice vectorial listo para retrieval efectivo:

    PREFLIGHT → LOAD → CLEAN → TAG → CHUNK → EMBED → SCORING → DEDUP → INDEX → INFORME

- **Preflight de disponibilidad** (en `embed.py`, fase `PREFLIGHT` antes de `LOAD`):
  comprueba que `EMBED_MODEL` esté disponible en `EMBED_PROVIDER` (Ollama `/api/tags`,
  HuggingFace `repo_exists`, Google listado) y **aborta** si no lo está; si no se puede
  verificar online (red cortada / sin API key) degrada a caché offline `EMBED_DIM_MAX_*`
  con aviso al usuario.
- **Núcleo clásico** (`load.py`, `clean.py`, `chunk.py`, `embed.py`, `index.py`, capa CSV):
  carga PDF/TXT/MD/CSV, normaliza el texto, trocea (actual: Level 2 recursive), genera
  embeddings multi-proveedor y los guarda en ChromaDB (métrica coseno).
- **Capa CSV** (`csv_advisor.py` + `csv_transform.py`): clasificación diferencial de cada CSV
  (entidad / hecho / textual / desconocido) que decide su tratamiento y su `dedup_policy`;
  la deduplicación de CSVs que **nunca es semántica**. Detalle en punto 2 (módulos) y en
  [`ANEXO_FUNC.md`](./ANEXO_FUNC.md), en punto 3 (funciones, tabla de decisiones del corpus 
  y glosario de logs).
- **TSD** (`tsd/tag.py`, `tsd/scoring.py`, `tsd/dedup.py`): etiquetado semántico con LLM
  (taxonomía cerrada), scoring de chunks (relevancia + centralidad + redundancia +
  autoridad) y deduplicación greedy por similitud coseno. El objetivo es que al índice
  entren solo chunks bien puntuados y no redundantes, para que el retrieval sea más
  preciso y la respuesta más fiable.
- **Multi-proveedor**: embeddings y LLM pueden salir de **Ollama** (offline, sin coste),
  **HuggingFace** (sentence-transformers + transformers) o **Google** (Gemini REST).
  Se cambia por variable de entorno (`EMBED_PROVIDER`, `GEN_PROVIDER`), sin tocar código.
- **Consola**: cada fase loguea su progreso y métricas (documentos, chunks, distribución
  de longitudes min/p25/media/p75/max, dims de embedding, scores mín/media/máx, chunks
  descartados por dedup, tiempo total).

### 1.1 Cómo lanzar todo el pipeline

Se puede lanzar el pipeline completo (`LOAD → INDEX`) desde de directorio raíz del proyecto: `python -m src.pipeline data --recreate-index`

> ◬ Previamente es preciso generar el entorno virtual de Python e instalar `pip install -r ./requirements.txt`.

Realmente admite algunos parámetros por encima de lo indicado en `.env` que he podido preparar:
```powershell
python.exe -m src.pipeline [-h] [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP] [--persist-dir PERSIST_DIR] [--collection COLLECTION] [--recreate-index] rutas [rutas ...]
```


## 2. Módulos

### `src/load.py` — carga
- Loaders por extensión: `.pdf` (PyPDFLoader), `.txt`/`.md` (TextLoader), `.csv` (una fila = un documento `k: v | k: v`).
- `cargar_archivos(rutas)`: archivo o carpeta (recursiva); devuelve `list[Document]` con `metadata.source` siempre poblado.

### `src/clean.py` — limpieza
- Normaliza antes de trocear: saltos de línea dobles, espacios sobrantes, líneas vacías, BOM y caracteres de control. No toca metadata.

### `src/chunk.py` — troceado (Level 2)
- `RecursiveCharacterTextSplitter` con separadores `["\n\n", "\n", ". ", " ", ""]`: primero límites semánticos (párrafo → frase), luego espacios.
- `trocear(documentos, chunk_size, chunk_overlap)`: acepta `str` (tests/scripts). Propaga la metadata del documento padre a cada chunk (`chunk_index` secuencial **por documento**). Es el **puente TAG→CHUNK**: `doc_category`, `tags`, `relevancia_llm` y los booleanos `tag_<nombre>` pasan automáticamente a los chunks.
- `chunk_index` es 0..n por fuente, no global (el id global lo asigna `pipeline.py`, §4).

### `src/embed.py` — vectores (multi-proveedor)
- `embeddear(textos)` según `EMBED_PROVIDER`:
  - **ollama**: endpoint batch `/api/embed` por lotes de `EMBED_BATCH_SIZE` (timeout 120 s).
  - **huggingface**: `SentenceTransformer` (carga perezosa + caché), `normalize_embeddings=True` (coherente con la métrica coseno).
  - **google**: REST `batchEmbedContents` (requiere `GOOGLE_API_KEY`).
- `embeddear_consulta(pregunta)`: fase online. **Condición RAG ineludible**: índice y consulta usan el mismo modelo.
- `exportar_json(chunks, ruta)`: persiste `text+metadata+embedding` para inspección/debug (`output/embeddings.json`).

### `src/index.py` — ChromaDB
- Cliente persistente, colección con métrica coseno. `indexar(...)` sanea metadatos (Chroma no acepta `None` ni listas), inserta y verifica `count() == len(ids)`. Soporta `recreate=True`.

### `src/tsd/tag.py` — etiquetado (LLM)
- Etiqueta **una vez por fuente** (1 llamada LLM por documento fuente, no por página) y propaga a todos sus documentos.
- Taxonomía cerrada: `doc_category = {tarifas, normativa, reservas, abonos, instalaciones, agenda}`, `tags` (máx. 8 de una lista de 16, como `str` porque Chroma no acepta listas) y `relevancia_llm` 0-1. El parse filtra a `CATEGORIAS_VALIDAS`/`TAGS_VALIDAS` (valores fuera de taxonomía se descartan o caen a `instalaciones`).
- **Booleanos por tag** `tag_<nombre>: True` (sparse, solo se escribe el tag presente): contrato de filtrado de la fase online — Chroma `where` soporta booleanos y la clave ausente no matchea (p. ej. `where={"$and": [{"doc_category": "reservas"}, {"tag_cancelación": True}]}`).
- Proveedor configurable (`GEN_PROVIDER`): Ollama / HuggingFace / Google.
- Robustez: si el LLM devuelve JSON malformado, asigna `doc_category="instalaciones"` por defecto en vez de romper el pipeline.

### `src/tsd/scoring.py` — scoring semántico
- `semantic_score = 0.45·relevancia_LLM + 0.20·centralidad + 0.20·(1 − redundancia) + 0.15·autoridad`, señales aditivas en [0, 1] (recortado y redondeado a 4 decimales) guardadas en `metadata['semantic_score']`.
- Pesos y justificación:

  | Componente | Peso | Qué mide | Por qué ese peso |
  |---|---|---|---|
  | `relevancia_llm` | 0.45 | Relevancia 0–1 que el LLM asigna a la fuente en `[TAG]` (¿responde a preguntas de tarifas, reservas, horarios y normas?). | Pesado dominante: es la única señal alineada directamente con el objetivo de RAG, no solo con la geometría de los embeddings. |
  | `centralidad` | 0.20 | Coseno del chunk contra el centroide de la colección (media de todos los vectores, normalizada). | Recompensa el contenido representativo del corpus y penaliza lo marginal, sin llegar a dominar el score. |
  | `1 − redundancia` | 0.20 | `1 − cos` con el vecino más cercano (FAISS `k=2`, excluyendo el propio). | Penaliza chunks casi duplicados: si el contenido ya existe en otro chunk, no aporta señal nueva. |
  | `autoridad` | 0.15 | Peso por tipo de fuente (substring en `source`): reglamento/normativa 1.0, precios/tarifas 0.9, agenda 0.6, resto 0.7. | Ajuste fino: una norma vale más que un CSV genérico, pero no puede compensar una baja relevancia. |

- El score alimenta el **dedup greedy** de `DEDUP` y, en el roadmap, el rerank de retrieval.
- Usa **FAISS** (`IndexFlatIP`, búsqueda k=2) para la redundancia en vez de materializar `E @ E.T`: memoria O(n·d) en vez de O(n²) (~1,3 GB en vez de ~68 GB en la escala real), que era la causa del OOM en la fase `[TSD]`.

### `src/tsd/dedup.py` — deduplicación
- Por `semantic_score` descendente: descarta un chunk si su coseno con algún chunk ya conservado es mayor o igual al `DEDUP_UMBRAL` (0.93 para all-minilm-l6-v2; subir a 0.95 si se dedup demasiado, bajar a 0.90 si queda redundancia).
- Implementación incremental con **FAISS** (`IndexFlatIP`): cada candidato se busca contra los ya conservados (k=1), en vez de reconstruir la matriz completa de similitud (misma OOM que scoring). Semántica idéntica a la matriz; tiempo acotado por el nº de chunks conservados.

### `src/pipeline.py` — orquestador
- `ejecutar_pipeline(rutas, ...)`: encadena todo y devuelve un dict de métricas: `num_documentos`, `num_chunks_pre_dedup`, `num_chunks_post_dedup`, `chunks_descartados`, `dim_embedding`, `chunk_stats` (min/p25/media/p75/max/cortos), `scoring`, `dedup` (incluye `chunks_por_categoria` de las 6 cerradas pre/post y `tags_top3_post`), `indice`, `tiempo_total_s`.

## 3. Proveedores y configuración (`.env`)

Para configurar un proveedor (EMBEDDINGS, TAGGING o GENERATION) basta con seleccionar uno de los tres para cada caso y tener su sección de modelos alimentada con los seleccionados. 

> `<PROVEEDOR>_TAG_MODEL`: Puede ser rellenado o no, en caso de no serlo asumirá el modelo de generación que esté seleccionado.

| Variable | Default (config.py) | Uso |
|---|---|---|
| `EMBED_PROVIDER` | `ollama` | `ollama` / `huggingface` / `google` |
| `TAG_PROVIDER` | hereda de `GEN_PROVIDER` (vacío = misma línea) | `ollama` / `huggingface` / `google` |
| `GEN_PROVIDER` | `ollama` | `ollama` / `huggingface` / `google` |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama local |
| `OLLAMA_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding Ollama |
| `OLLAMA_GEN_MODEL` | `llama3.2` | LLM Ollama |
| `HF_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding HuggingFace |
| `HF_GEN_MODEL` | `mistralai/Mistral-7B-v0.1` | LLM HuggingFace |
| `HF_DEVICE` | `cpu` | `cpu` / `cuda` |
| `HF_TOKEN` | — | solo modelos gated |
| `GOOGLE_API_KEY` | — | embeddings/LLM Google |
| `GOOGLE_EMBED_MODEL` | `gemini-embedding-2` | embedding Google |
| `GOOGLE_GEN_MODEL` | `gemini-2.0-flash` | LLM Google |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | troceado |
| `EMBED_DIM` | `384` | cap de dimensión del índice: `dim final = min(dim del modelo, EMBED_DIM)` (ver `src/embed.py`) |
| `EMBED_BATCH_SIZE` | `30` | lotes de embedding |
| `EMBED_DIM_MAX_OLLAMA` / `_HF` / `_GOOGLE` | (vacío) | caché offline del preflight: dim máxima del modelo cuando no se puede verificar online; vacío = no declarado |
| `EXPORT_EMBEDDINGS` | `true` | volca `output/embeddings.json` al terminar |
| `TAG_SCORING_DEDUP` | `true` | activa/desactiva TSD |
| `DEDUP_UMBRAL` | `0.93` | umbral de dedup |
| `CHROMA_DIR` / `COLLECTION_NAME` | `./output/chroma_db` / `deporte_municipal` | índice |

Los parámetros de retrieval (`TOP_K`, `MAX_CHUNKS`) son de la fase online pendiente (§7) y aún no viven en `config.py`.

## 4. Flujo de invocación (quién llama a quién)

| Invocador | Funciones que llama | Con qué |
|---|---|---|
| `pipeline.py::ejecutar_pipeline` | `cargar_archivos` → `limpiar` → `etiquetar` → `trocear` → `embeddear` → `puntuar` → `deduplicar` → `indexar` | constantes de `config.py` por omisión |
| `tag.py::etiquetar` | chat del proveedor (Ollama / HF / Google) | documentos limpios con `metadata.source` |
| `scoring.py::puntuar` | FAISS (`IndexFlatIP`) + numpy | chunks + embeddings (requiere tag previo) |
| `dedup.py::deduplicar` | FAISS (`IndexFlatIP`) + numpy | chunks + embeddings (requiere scoring previo) |
| `[retrieval]` (pendiente) | `embeddear_consulta` + `obtener_cliente_chroma` | `config.COLLECTION_NAME` |

Dependencia por `metadata`: **load → tag → scoring → dedup → index**. Cada módulo asume que el anterior corrió:
- `tag.py` escribe `doc_category`, `tags`, `relevancia_llm` y los booleanos `tag_<nombre>` (por fuente).
- `chunk.py` los propaga a cada chunk junto con `chunk_index`.
- `scoring.py` lee `relevancia_llm` y escribe `semantic_score`.
- `dedup.py` lee `semantic_score`.
- `index.py` sanea todo (Chroma no acepta `None` ni listas).

## 5. Métricas por consola (en tiempo de ejecución)

- `[LOAD]` nº documentos cargados.
- `[CLEAN]` nº documentos normalizados.
- `[TAG]` por fuente: categoría + relevancia; al final: nº fuentes, tiempo y proveedor.
- `[CHUNK]` nº chunks y distribución de longitudes: min / p25 / media / p75 / max + nº de chunks cortos (<50 chars).
- `[EMBED]` nº vectores y dimensión.
- `[SCORE]` scores mín / media / máx + mejor chunk.
- `[DEDUP]` umbral usado y nº/porcentaje de chunks descartados.
- `[INDEX]` nº vectores en la colección.
- `[FIN]` tiempo total.

El dict devuelto por `ejecutar_pipeline` incluye las mismas métricas para eval.

## 6. Text splitting: evaluación (5 niveles)

He creado un ANEXO para saber cómo modificar los niveles de chunking en el proyecto. Para más información [consultar ANEXO](./ANEXO_CHUNKING.md).

| Nivel | Método | Estado |
|---|---|---|
| 1 | Character (fijo) | No usado: rígido, ignora la estructura. |
| 2 | Recursive character | **Actual**. Separadores párrafo → frase → espacio → letra; recomendado como punto de partida. |
| 3 | Document-specific | Oportunidad futura: `MarkdownTextSplitter` para `.md`; tablas de PDF con Unstructured si el corpus gana tablas. |
| 4 | Semantic (breakpoints por embeddings) | Caro. Solo si el eval de retrieval muestra que el Level 2 no basta. |
| 5 | Agentic (LLM decide) | No recomendado: lento y caro. |

Decisión: mantener Level 2 y validarlo con `scripts/eval_coherencia_chunks.py` ("Chunking Commandment": el objetivo es que el dato se recupere con valor).

## 7. Fase online (retrieval) — pendiente

- `embeddear_consulta` + `obtener_cliente_chroma` + **filtro por `doc_category` y booleanos `tag_<nombre>`** (clasificación de intención por palabras clave; `where` de Chroma sobre los booleanos que ya están en el índice) + rerank por `semantic_score`.
- Con filtro + rerank, `TOP_K` podría subir de 3 a 5 sin perder precisión.