# ANEXO_FUNC — Funciones de cada archivo

Guía de referencia de las funciones de todos los archivos del proyecto (fuente, scripts, tests y datos), para orientar sobre consultas tipo «¿dónde está X?». Las firmas corresponden al código actual; la documentación de conceptos (scoring, dedup por política, dimensión garantizada, informes) vive en los demás documentos del [`proyecto`](../README.md#documentación-técnica).

> Convención: se listan las funciones **públicas** (las que entran por importar). Las privadas (prefijo `_NOMBRE`) se mencionan solo cuando son parte esencial del comportamiento.

## 1. Raíz del proyecto

### `config.py`

Constantes del proyecto, leídas de `.env` con defaults: se cambia de proveedor sin tocar el código. No expone funciones de negocio; su «API» es el conjunto de constantes:

| Constante | Default | Efecto |
|---|---|---|
| `EMBED_PROVIDER` | `ollama` | Proveedor de embeddings (`ollama` / `huggingface` / `google`) |
| `EMBED_MODEL` (resuelta) | según proveedor | Modelo efectivo de embeddings (la usan `src/embed.py` y la consulta) |
| `GEN_PROVIDER` / `GEN_MODEL` | `ollama` / llama3.2 | Proveedor y modelo de generación (fase online futura) |
| `TAG_PROVIDER` / `TAG_MODEL` | hereda de `GEN_*` | Proveedor y modelo del etiquetado; si en `.env` están vacíos, heredan los de `GEN` |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Endpoint de Ollama (batches `/api/embed` y `/api/chat`) |
| `HF_DEVICE` | `cpu` | `cpu` / `cuda` para modelos HuggingFace (`HF_TOKEN` opcional, solo gated) |
| `GOOGLE_API_KEY` | — | Clave API; obligada si el proveedor `google` se usa en alguno de los interruptores |
| `OLLAMA_*_MODEL`, `HF_*_MODEL`, `GOOGLE_*_MODEL` | ver `.env.example` | Modelos por proveedor de cada interruptor (EMBED / GEN / TAG) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Longitud y sobrelap del troceado |
| `EMBED_DIM` | `384` | Tope máximo de dimensión: `dim índice = min(dim modelo, EMBED_DIM)` (ver `src/embed.py`) |
| `EMBED_BATCH_SIZE` | `30` | Tamaño de lote por proveedor |
| `EMBED_DIM_MAX_OLLAMA` / `_HF` / `_GOOGLE` | `None` | Caché offline del preflight: dim máxima que maneja el modelo cuando la comprobación online no es posible (red cortada / sin API key) |
| `EXPORT_EMBEDDINGS` | `true` | Exporta `output/embeddings.json` al terminar |
| `TAG_SCORING_DEDUP` | `true` | Activa el bloque TSD completo (TAG + SCORING + DEDUP) |
| `DEDUP_UMBRAL` | `0.93` | Umbral coseno de deduplicación semántica |
| `CHROMA_DIR` | `./output/chroma_db` | Directorio persistente del índice |
| `COLLECTION_NAME` | `deporte_municipal` | Nombre de la colección Chroma |
| `COSINE_SPACE` | `cosine` | Métrica de la colección (coherente con embeddings normalizados) |

Función interna única: `_resolver(provider, llm_modelos, switch de provider)` — resuelve el modelo efectivo de cada interruptor según su proveedor y lanza `ValueError` si el proveedor no soporta el modelo indicado.

### `__init__.py` (raíz)

Todavía vacío: hace el directorio paquete (permitir `python -m src.pipeline` desde la raíz).

## 2. Paquete `src/` (núcleo del pipeline)

### `src/__init__.py`

Exposición del paquete con **carga perezosa** (`__getattr__` + `importlib`): llamado así porque las dependencias y submódulos no se cargan al ejecutar `import src`, sino únicamente cuando se accede por primera vez al atributo correspondiente. Así no se arrastran de entrada dependencias pesadas (Chroma, sentence-transformers, FAISS); cada submódulo (`load`, `clean`, `chunk`, `embed`, `index`, `pipeline`, `csv_advisor`, `csv_transform`) se importa bajo demanda de la función que proceda.

### `src/pipeline.py` — orquestador

Punto de entrada del pipeline completo (`PREFLIGHT → LOAD → CLEAN → TAG → CHUNK → EMBED → SCORING → DEDUP → INDEX → INFORME`), siempre con el switch del módulo `tsd` activado (ver README.md)[../README.md].

| Símbolo | Descripción |
|---|---|
| `ejecutar_pipeline(rutas, *, chunk_size, chunk_overlap, persist_dir, collection_name, recreate_index, export_embeddings, informe) -> dict` | Encadena todas las fases, loguea el avance por consola y devuelve un dict de métricas (documentos, chunks pre/post dedup, stats de longitudes, dim, scoring, dedup, índice, tiempos por fase). Genera `output/informe_index_aaaaMMdd_hhmm.md` salvo `informe=False` (tests). |
| `__main__` (argparse) | CLI: `python -m src.pipeline [-h] [--chunk-size N] [--chunk-overlap N] [--persist-dir D] [--collection N] [--recreate-index] rutas...` |
| `_log(fase, msg)` | Marca de tiempo + fase en consola (formato único). |
| `_comprobar_dim(dim, embed_dim, dim_max_modelo) -> (nivel, msg)` | Compara la dim resultante con `EMBED_DIM` y la dim máxima declarada: devuelve `vacio` (corpus vacío), `ok`, `info` (`EMBED_DIM` < dim modelo) o `aviso` (`EMBED_DIM` > dim modelo, en cuyo caso el pipeline ajusta `config.EMBED_DIM` a la dim real). |
| `_stats_chunks(chunks) -> dict` | Distribución de longitudes de chunk: `min`, `p25`, `media`, `p50`, `p75`, `max` y `cortos` (< 50 chars = ruido probable). |

Comportamiento clave:

- El id de cada chunk es **global y único** (`{source}::{posición global}`): `chunk_index` es secuencial por documento y un CSV produce muchos documentos con la misma fuente; un id no global haría que Chroma sobreescribiera filas.
- Si `TAG_SCORING_DEDUP=false`, se omiten TAG, SCORING y DEDUP (el pipeline sigue siendo correcto: el dedup solo se saltea si TSD está apagado también).
- El dict devuelto contiene las claves que consume `src/informe.py` (`parametros`, `fases`, `resumen_fases`, `preflight`, `scoring`, `dedup`, `indice`): es el contrato entre pipeline e informe.

### `src/load.py` — carga de corpus

Abstrae el formato: PDF, TXT, MD, CSV → `list[Document]`.

| Símbolo | Descripción |
|---|---|
| `cargar_archivos(rutas: list[str] \| str) -> list[Document]` | Punto de entrada público. Recibe una ruta o lista de archivos/carpetas; recorre carpetas de forma recursiva (`rglob`), aplica el loader de cada extensión y **siempre** asegura `metadata.source` (basename) aunque el loader no lo haya puesto. Los archivos con extensión no soportada se ignoran con aviso en consola. |
| `_detectar_encoding(path) -> str` | Detecta la encoding probando `utf-8-sig` → `cp1252` → `latin-1` (la última nunca falla). |
| `_load_pdf(path)`, `_load_text(path)`, `_load_csv(path)` | Loaders por extensión (registro `dict[str, Loader]` `_LOADERS`: `.pdf` PyPDFLoader, `.txt`/`.md` TextLoader, `.csv` delega en `csv_transform`). |

Comportamiento clave: `load.py` es solo un **lector**; todo el tratamiento diferencial del CSV se delega en `csv_transform.transform_csv` (decidido por `csv_advisor`).

### `src/clean.py` — normalización de texto

| Símbolo | Descripción |
|---|---|
| `limpiar(documentos: list[Document] \| str) -> list[Document] \| str` | Aplica `_normalizar_page_content` a cada documento (o al string, para tests) y devuelve una copia con los metadatos intactos (copia shallow de `metadata`). No muta los objetos recibidos. |
| `_normalizar_page_content(texto) -> str` | Reglas en orden: saltos de línea múltiples → uno; espacios/tabs múltiplos → uno; líneas vacías repetidas; strip de inicio/final por línea; eliminación de caracteres de control y BOM. |

### `src/chunk.py` — troceado

| Símbolo | Descripción |
|---|---|
| `trocear(documentos: list[Document] \| str, chunk_size=None, chunk_overlap=None) -> list[Document]` | Divide con `RecursiveCharacterTextSplitter` (separadores `["\n\n", "\n", ". ", " ", ""]`: párrafo → frase → espacio → letra, el «Level 2» recomendado como punto de partida). Acepta string plano (tests/eval). Añade a la metadata `chunk_index` (secuencial **por documento**) y `chunk_size`. Es el **puente TAG→CHUNK**: propaga `doc_category`, `tags`, `relevancia_llm` (y las claves de CSV) del documento padre a cada chunk. |

Comportamiento clave: respeta `chunking_hint` del advisor:

- `no_chunk` → el documento queda como único chunk (entidad atómica, p. ej. una piscina).
- `light_chunk` → si cabe en `chunk_size`, también queda sin trocear.
- En caso contrario → el splitter normal.

La comprobación de defaults es `is not None`, así que `chunk_overlap=0` es un valor válido (no se traga como falsy).

### `src/embed.py` — vectores (multi-proveedor)

Única puerta de conversión texto → vectores: índice y consulta futura pasan por aquí, por lo que la **dimensión se garantiza una sola vez**.

| Símbolo | Descripción |
|---|---|
| `embeddear(textos, model=None, dim=None) -> list[list[float]]` | Convierte textos en vectores con el proveedor de `EMBED_PROVIDER`. Envía `dim` como pista (Ollama `dimensions` payload, HF `truncate_dim`) y aplica `_corte_dim` como garantía real: nunca devuelve más de `dim` dimensiones (corte del prefijo + renormalización). |
| `embeddear_consulta(pregunta, model=None) -> list[float]` | Embed de una sola consulta (fase online futura). Misma garantía de dim y **mismo modelo** que al indexar (condición RAG). |
| `exportar_json(chunks, embeddings=None, ruta="output/embeddings.json")` | Persiste `text + metadata + embedding` por registro (inspección/debug; lo consume `scripts/benchmark_tsd.py`). |
| `verificar_modelo_disponible(metadatos=None) -> PreflightResultado` | **Preflight**: comprueba que `EMBED_MODEL` está disponible en `EMBED_PROVIDER` **antes** de empezar. Ollama (`/api/tags`), HuggingFace (`repo_exists`), Google (listado v1beta). Si el modelo no está disponible: `RuntimeError` (el pipeline debe abortar). Si no se puede comprobar online (red cortada / Ollama apagado / sin API key): cae al caché `.env` `EMBED_DIM_MAX_<proveedor>` y devuelve `verificado_por="cache_env"` con aviso. |
| `_embed_ollama / _embed_huggingface / _embed_google` | Implementaciones por proveedor (batch `/api/embed` por lotes de `EMBED_BATCH_SIZE`; `SentenceTransformer` con caché y `normalize_embeddings=True`; REST `batchEmbedContents`). |
| `_normalizar(vectores)`, `_corte_dim(vectores, dim_cap)` | Renormalización por norma euclídea (evita división por cero) y recorte de prefijo a `dim_cap` + renormalización (coherente con la métrica coseno de Chroma). |

Comportamiento clave: registro `_PROVIDERS` — añadir un proveedor nuevo es añadir una función y una línea al dict.

### `src/index.py` — ChromaDB

| Símbolo | Descripción |
|---|---|
| `obtener_cliente_chroma(persist_dir=None) -> chromadb.ClientAPI` | Cliente persistente (`PersistentClient`) en `CHROMA_DIR`. Reutilizable por la futura fase de retrieval. |
| `obtener_guid_chroma(persist_dir=None) -> str | None` | Devuelve el GUID de la carpeta de datos de ChromaDB en `persist_dir`/`CHROMA_DIR` (o `None` si no existe). `pipeline.py` lo usa para nominar el informe (`informe_index_<GUID8>_...md`) y detectar el índice existente. |
| `crear_coleccion(client, nombre=None)` | Crea o recupera la colección con `hnsw:space = cosine` (coherente con embeddings normalizados). |
| `indexar(ids, embeddings, documents, metadatos, persist_dir=None, collection_name=None, recreate=False) -> (vivos, totales)` | Inserta por lotes (`_BATCH_ADD = 2000`, el límite real del backend varía con la dim), sanea los metadatos (Chroma no acepta `None` ni listas: las convierte a `"null"` / `str`), soporta `recreate` y **verifica** la inserción (batch por batch contra `collection.get`) antes de devolver la cifra. `totales` es `collection.count()`, así que un re-index parcial queda visible. |
| `_sanear(meta)` | Sanitizado de metadatos para Chroma. |

Comportamiento clave: la verificación es por lotes porque un `get(ids=...)` masivo estalla en «too many SQL variables»; el assert de ids únicos protege contra ids duplicados (ver `src/pipeline.py`).

### `src/informe.py` — informe de indexación

Renderiza el dict de métricas del pipeline en markdown (`output/informe_index_aaaaMMdd_hhmm.md` por defecto). **Solo renderiza**: no lee config ni hace side effects (testea aislada).

| Símbolo | Descripción |
|---|---|
| `generar_informe(datos, ruta=None) -> Path` | Escribe el informe (7 secciones: parámetros · preflight · métricas por fase · chunking · índice · TSD · señales) en `ruta` (por defecto `output/informe_index_aaaaMMdd_hhmm.md`, con fecha/hora locales) y devuelve la ruta. |
| `formatear_duracion(segundos) -> str` | Duración legible: `—` para `None` / < 0.05 ms (evita falsos 0), `X ms` para < 1 s, `X.XX s` para el resto. |
| `_tabla(filas)`, `_fases_md(fases, resumen)`, `_scoring_md(scoring, dedup)`, `_senales(datos)` | Bloques internos de renderizado (parámetros, fases, TSD, heurísticas automáticas de decisión). |

Las condiciones de las señales (condición → acción) y las reglas de la tabla de parámetros están documentadas en [`ANEXO_OUTPUTS.md`](./ANEXO_OUTPUTS.md).

## 3. Capa CSV (clasificación diferencial)

### `src/csv_advisor.py`

Decide el **tratamiento** de cada CSV antes de trocear, con dos niveles de decisión (de lo barato a lo caro, sin embeddings):

| Símbolo | Descripción |
|---|---|
| `advise_csv(path) -> (CsvAdvice, CsvProfile)` | Punto de entrada: receta de fuente conocida (`RECETAS_CONOCIDAS`, confianza 0.99) o, si el esquema cambió / fuente desconocida, heurística de perfil estructural. Nunca devuelve `None`. |
| `inspect_csv(path, sample_size=2000) -> CsvProfile` | Perfil estructural sobre una muestra: columnas ID/medida/temporal/narrativa (por tokens en el nombre), densidad numérica real por columna, cardinalidades, ratio de unicidad de filas. |
| `match_known_source(profile) -> CsvAdvice \| None` | Receta fija para fuentes conocidas de `datos.madrid.es`; `None` si la fuente no es conocida o ya no contiene las columnas esperadas (entonces manda a la heurística). |
| `infer_csv_kind(profile) -> CsvAdvice` | Heurística por perfil: 1) entidad (ID estable, sin temporal) → `entity_doc` + `exact_key`; 2) hecho/serie temporal (medidas numéricas + ≥3 dimensiones) → `grouped_doc` + `group_only`; 3) textual → `row_as_doc` + `semantic_optional`; 4) fallback conservador `unknown_csv` + `exact_only`. |
| `leer_filas_csv(path, sample_size=None) -> (columnas, filas)` | Lectura robusta compartida: encoding + delimitador (`;`/`,`/tab) detectados, claves basificadas (`MXASSETNUM,C,12` → `mxassetnum`), alias `§`→`º` (el ordinal del corpus sale corrompido según encoding). |
| `detectar_encoding`, `normalizar_col`, `detectar_delimitador` | Utilidades de lectura (delimitador = el que produce un nº estable de columnas entre cabecera y datos). |

Tipos: `CsvProfile` (métricas estructurales) y `CsvAdvice` (decisión ejecutable: `csv_kind`, `treatment`, `dedup_policy`, `id_columns`, `measure_columns`, `grouping_keys`, `confidence`, `reason`).

Decisión aplicada al corpus actual (`data/`), por receta conocida o por heurística:

| Fuente | Columna ID | `csv_kind` | Tratamiento | Dedup |
|---|---|---|---|---|
| `200186-0-polideportivos.csv` | `pk` | `entity_table` | `entity_doc` | `exact_key` |
| `200215-0-instalaciones-deportivas.csv` | `pk` | `entity_table` | `entity_doc` | `exact_key` |
| `210227-0-piscinas-publicas.csv` | `pk` | `entity_table` | `entity_doc` | `exact_key` |
| `300390-0-areas-deportivas.csv` | `mxassetnum` | `entity_table` | `entity_doc` | `exact_key` |
| `212504-0-agenda-actividades-deportes.csv` | `id-evento` | `entity_table` | `entity_doc` | `exact_key` |
| `300085-0-deportes_abonos.csv` | (sin ID) | `fact_table` | `grouped_doc` | `group_only` |
| `300097-0-deportes-descuentos.csv` | (sin ID) | `fact_table` | `grouped_doc` | `group_only` |

- **Tablas de entidad** (centros, instalaciones, piscinas, áreas, eventos): cada fila es una entidad con **identificador estable** → se documenta como entidad (`entity_doc`) y se deduplica **por clave exacta** (`exact_key`). La dedup semántica se evita porque dos filas casi iguales (p. ej. dos piscinas del mismo barrio) son entidades distintas, no duplicados.
- **Tablas de hechos** (abonados, descuentos): sin columna ID; cada fila es un registro repetitivo, por tanto se agrupan por dimensiones (`grouped_doc`) y solo se deduplican los grupos repetidos (`group_only`), no compiten entre sí en la deduplicación semántica.

**Glosario de los logs:**

- **"PK estable" vs. "identificador estable"**: misma razón en `RECETAS_CONOCIDAS` con matiz cosmético distinto: se usa "PK" cuando la columna se llama literalmente `pk` y "identificador estable" cuando el ID tiene otro nombre (`mxassetnum`, `id-evento`). No es un concepto distinto.
- **"Índice"**: solo existe el **índice vectorial de Chroma** (`CHROMA_DIR`): no hay un "índice estable": si cambias el modelo de embeddings o el chunking, el índice se regenera con `--recreate-index`.

### `src/csv_transform.py`

Materializa el `CsvAdvice`: filas → `Document` enriquecidos.

| Símbolo | Descripción |
|---|---|
| `transform_csv(path) -> list[Document]` | Punto de entrada (lo llama `load.py`). Emite el log `[CSV]` con la decisión (kind · treatment · dedup · confianza) y aplica la estrategia por `treatment`. |
| `entidad_a_documento(source, row, row_i, columnas, advice)` | Una fila de entidad → un documento (`dedup_policy=exact_key`, `entity_key` desde la columna ID, `chunking_hint=no_chunk`, + `district`/`neighborhood` si existan). |
| `grupo_a_documento(source, keys, rows, columnas, advice)` | Un grupo de filas de hecho (agrupado por `grouping_keys`) → un documento agregado con el desglose línea a línea (`dedup_policy=group_only`, `is_aggregated=True`, `chunking_hint=light_chunk`). |
| `filas_a_documentos(source, rows, columnas, advice)` | Fila por documento plano (`row_as_doc`: tablas textuales o `unknown_csv`). |
| `limpiar_valor(valor) -> str` | Compacta espacios consecutivos de un valor de celda (usado al materializar filas). |

Comportamiento clave: la metadata (`dedup_policy`, `entity_key`, `group_key`, `chunking_hint`) viaja con el chunk hasta el índice y gobierna la fase DEDUP (la dedup de CSVs **nunca es semántica**, sino por clave exacta).

## 4. Paquete `src/tsd/` (Tagging, Scoring, Dedup)

Etiquetado semántico + scoring + deduplicación para que al índice entré señal limpia. Cada una de las tres funciones escribe en la metadata que consume la siguiente (TAG → CHUNK → SCORING → DEDUP).

### `src/tsd/tag.py` — etiquetado con LLM

| Símbolo | Descripción |
|---|---|
| `etiquetar(documentos) -> list[Document]` | Pregunta al LLM **una vez por fuente** (no por página/fila) y propaga `doc_category`, `tags` (string con `;` porque Chroma no acepta listas; lectura humana), `relevancia_llm` (0–1) y **booleanos por tag** `tag_<nombre>: True` (sparse) a todos los documentos de esa fuente. Taxonomía cerrada: `tarifas`, `normativa`, `reservas`, `abonos`, `instalaciones`, `agenda` + 16 tags (`abono, piscina, reserva, tarifa, horario, empadronado, descuento, competición, instalación, precio, cancelación, devolución, accesibilidad, aire_libre, inscripción, temporada`). Los booleanos son el contrato de filtrado de la fase online (`where` de Chroma). Si el LLM devuelve JSON malformado, marca por defecto (`instalaciones`, relevancia 0.5) en vez de romper el pipeline. |
| `CATEGORIAS_VALIDAS` / `TAGS_VALIDAS` | `frozenset` de la taxonomía cerrada: el parse filtra a estos valores (categoría fuera → `instalaciones` con aviso en consola; tags fuera → se descartan silenciosamente). Garantiza que la metadata del índice sigue siendo filtrable en la fase online. |
| `_chat_ollama / _chat_huggingface / _chat_google` | Chat por proveedor (Ollama `/api/chat`; HF `pipeline("text-generation")` con caché; Google REST `generateContent`). Registro `_CHAT` (añadir proveedor = +1 función + 1 línea). |
| `_parse_json(texto)` | Extrae el JSON de la respuesta del LLM (tolera bloques `json` y prosa previa/trasera); lanza `JSONDecodeError` si no hay objeto válido. |

Nota de coste: el prompt solo ve los **primeros 6000 caracteres** de cada fuente (`PROMPT` + `[:6000]`); es el compromiso de coste por fiabilidad (fuente etiquetada de forma consistente). La duración y el proveedor/modelo usados se loguean en consola.

Nota de filtrado (fase online): `tags` se guarda como string unido por `;` (lectura humana; **no** se puede filtrar por substring con `where`, solo exact match del string entero). El filtrado real lo hacen los **booleanos por tag** que `etiquetar` escribe además: `tag_<nombre>: True` de forma *sparse* (solo se crea la clave del tag presente, p. ej. `tag_piscina: True`, `tag_cancelación: True`). Chroma `where` soporta booleanos y una clave ausente no matchea, así que la fase online (`--query` del roadmap) filtra directamente sobre el índice:

```python
# filtrado simple
where={"tag_piscina": True}
# filtrado compuesto: categoría + tag
where={"$and": [{"doc_category": "reservas"}, {"tag_cancelación": True}]}
```

Los booleanos se propagan con la metadata (TAG → CHUNK) y sobreviven al saneo de `index.py` (Chroma acepta `True`/`False`), por lo que cada chunk del índice es filtrable por categoría, por tag y por `semantic_score` (rerank).

### `src/tsd/scoring.py` — scoring semántico

| Símbolo | Descripción |
|---|---|
| `puntuar(chunks, embeddings, info=None) -> list[Document]` | Calcula `semantic_score = 0.45·relevancia_LLM + 0.20·centralidad + 0.20·(1 − redundancia) + 0.15·autoridad` (recortado a [0,1] y redondeado a 4 decimales) por chunk y lo guarda en `metadata['semantic_score']`. Vuelca métricas en `info` (mín/media/máx, centralidad y redundancia medias, % chunks ≥ 0.6, tiempo). |
| `_autoridad(source) -> float` | Peso por tipo de fuente (coincidencia por substring en `source`): reglamento/normativa 1.0, precios/tarifas 0.9, agenda 0.6, resto 0.7. |
| `_redundancias(E) -> np.ndarray` | Similitud coseno de cada vector con su vecino más cercano (excluido el propio) usando FAISS `IndexFlatIP` con búsqueda `k=2`: memoria O(n·d) en vez de O(n²) (~1,3 GB vs ~68 GB en la escala real n≈130k, d=2.560). `k=2` es suficiente porque la mejor similitud de una fila siempre está en su top-2. |

### `src/tsd/dedup.py` — deduplicación

| Símbolo | Descripción |
|---|---|
| `deduplicar(chunks, embeddings, umbral=DEDUP_UMBRAL, info=None) -> (chunks, embeddings)` | Dedup **consciente de política**: los chunks con `dedup_policy` exacta (`exact_only` / `exact_key` / `group_only`) se deduplican **solo por clave** (repetición literal) y **nunca por coseno**; los demás pasan por la deduplicación semántica greedy por `semantic_score` descendente. Vuelca métricas en `info` (umbral, pre/post, descartados exactos vs. semánticos, tiempo, `chunks_por_categoria` de las 6 cerradas pre/post y `tags_top3_post`). |
| `_deduplicar_semantico(chunks, embeddings, umbral) -> (índices_kept, embeddings_kept)` | Greedy incremental con FAISS: cada candidato se busca (k=1) contra el índice plano que va creciendo con los chunks ya conservados. Semántica idéntica a la matriz completa `sim[i,j] < umbral`, memoria O(n·d). |
| `_clave_exacta(chunk) -> tuple` | Clave de identidad por política: entidad (`source` + `entity_key`), grupo (`source` + `group_key` + `chunk_index`) o fila literal (`source` + `row` + contenido). |

## 5. Scripts de validación y experimentación

### `scripts/eval_coherencia_chunks.py`

Evalúa si los cortes son coherentes: trocea un texto de dominio repetido con `CHUNK_SIZE`/`CHUNK_OVERLAP` de `.env`, genera embeddings con el proveedor y compara la similitud coseno **media entre adyacentes** frente a la **media de pares aleatorios**. Margen > 0.05 → el overlap mantiene coherencia temática; ≈ 0 → cortes arbitrarios. Ejecución: `python -m scripts.eval_coherencia_chunks`.

### `scripts/benchmark_tsd.py`

Verifica a escala real el coste de TSD (scoring + dedup) tras la migración a FAISS: carga `output/embeddings.json` (requiere `EXPORT_EMBEDDINGS=true`) y ejecuta `puntuar` + `deduplicar` sin ChromaDB ni Ollama. Ejecución: `python -m scripts.benchmark_tsd`.

### `scripts/test_embeddings.py`

Test rápido de modelos de embedding en Ollama: 4 frases (3 relacionadas + 1 irrelevante), similitud coseno por pares y **margen** (relevante − irrelevante) como separador de dominio. Útil para elegir `OLLAMA_EMBED_MODEL`. Requiere Ollama arrancado con los modelos descargados. Ejecución: `python -m scripts.test_embeddings`.

## 6. Pruebas (`tests/`)

Suite pytest offline (providers fake / mocks; sin red ni `output/`):

| Test | Qué cubre |
|---|---|
| `test_load.py` | Detección de encoding (utf-8/cp1252/latin-1, con BOM y bytes corruptos) y loaders TXT/CSV: carga sin errores ni carácter fantasma. |
| `test_clean.py` | `limpiar`: BOM, colapso de espacios/lineas vacías, caracteres de control, preservación de contenido + métricas de compresión. |
| `test_chunks.py` | `trocear`: límites de tamaño, overlap entre adyacentes, metadata (`chunk_index`, `source`) y preservación de palabras/números (tarifas, horarios); métricas de chunking para el informe. |
| `test_embed.py` | Preflight (Ollama/HF/Google + caché offline `EMBED_DIM_MAX_*`), recorte `_corte_dim` y seguridad de dimensiones (`embeddear` / `_comprobar_dim`). |
| `test_csv_layers.py` | Capa CSV sobre las familias reales del corpus: profiling (encoding, cabeceras, medidas), recetas conocidas vs. heurística, transformación (entidad/grupo/fila) y dedup por `dedup_policy`. |
| `test_tsd.py` | `puntuar` (score, penalización por redundancia, autoridad normativa, acotado a 1.0) y `deduplicar` (conservar el de mejor score, umbrales, orden original, volcado de cobertura por categoría y top-3 tags) + verificación contra una referencia de matriz completa (300 vectores). |
| `test_informe.py` | `generar_informe`: renderizado del MD (parámetros, fases, chunking, TSD, 6.3 cobertura por categoría) y que `puntuar`/`deduplicar` vuelcan sus métricas en el dict `info`. |
| `test_informe_e2e.py` | Humo extremo a extremo offline (proveedor fake): `ejecutar_pipeline()` completo, dict de métricas completo, informe con sus 7 secciones y ChromaDB con los vectores esperados. |
| `test_preguntas.py` | `queries/preguntas.json`: snapshot estable (18, orden, ids), `categoria_esperada` ∈ taxonomía cerrada y `tags_esperados` ∈ `TAGS_VALIDAS` (la multi-aspecto nº8). |

Ejecución: `python -m pytest tests/ -v`.

## 7. Otros

| Archivo | Rol |
|---|---|
| `queries/preguntas.json` | 18 preguntas de evaluación del dominio (tarifas, abonos, reservas, agenda), cada una con `id`, `pregunta`, `categoria_esperada` (única: categoría primaria) y `fuente_esperada` (la fuente del corpus que debería dar el hit) y `tags_esperados` (lista ∈ `TAGS_VALIDAS`: espejo multi-aspecto de los tags del documento, para evaluar coincidencia de etiquetas en la fase online). |
| `data/` | Corpus: 8 PDF normativos/tarifas, 7 CSV de `datos.madrid.es` y 1 TXT de agenda. Los CSV pueden regenerarse descargando cualquiera de los datasets de [datos.madrid.es · grupo deporte](https://datos.madrid.es/group/deporte) (el loader los recarga igual aunque cambien de esquema ligero). |
| `output/` | Artefactos generados (gitignored): `chroma/` (índice persistente), `embeddings.json` (si `EXPORT_EMBEDDINGS=true`), `informe_index_aaaaMMdd_hhmm.md` (cada ejecución). |
| `.env` | Configuración local (gitignored). El template canónico es `.env.example`. |
| `config.py` | Ver §1. |
