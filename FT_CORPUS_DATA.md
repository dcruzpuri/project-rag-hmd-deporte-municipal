# FT_CORPUS_DATA — pipeline offline (feat/corpus-data)

**Dirigido a:** desarrollador de `FEAT/Retrieval+contexto` y siguientes
**Estado:** pipeline offline implementado. Módulos `tag`/`scoring`/`dedup` implementados (integración pendiente) 

Este documento describe **qué hace cada archivo y función**, **quién lo invoca**, **dónde va el resultado** y la **lógica de diseño** detrás de cada decisión, para que el módulo de retrieval se construya sobre un contrato estable.

---

## Arquitectura y flujo

```text
Documentos
    ↓
LOAD        load.py        (abstracción de formato)
    ↓
CLEAN       clean.py       (normalización)
    ↓
TAG         tag.py         (Llama4: taxonomía cerrada por fuente)   ← NUEVO
    ↓
CHUNK       chunk.py       (troceado con overlap)
    ↓
EMBED       embed.py       (Ollama /api/embed, batch)
    ↓
SCORING     scoring.py     (semantic_score 0-1)                     ← NUEVO
    ↓
DEDUP       dedup.py       (deduplicación semántica greedy)          ← NUEVO
    ↓
INDEX       index.py       (ChromaDB persistente, cosine)
    ↓
[ONLINE — pendiente FEAT/Retrieval+contexto]
RETRIEVE    retrieve.py    (top-k + where doc_category + rerank)
    ↓
GENERATE    generate.py    (prompt + abstención)
    ↓
CLI main.py · Streamlit app.py · logging_utils.py
```

**Separación offline/online** (requisito del enunciado): todo lo de arriba se ejecuta **una vez** y persiste en `output/chroma_db`. Lo online (retrieval/generación) es un proceso distinto que solo **lee** el índice.

---

### Invocación de archivos (quién llama a quién y funciones)

| Invocador | Funciones que llama | Con qué |
|---|---|---|
| `pipeline.py::ejecutar_pipeline` | `cargar_archivos` → `limpiar` → `trocear` → `embeddear` → `indexar` | constantes de `config.py` por omisión |
| `tag.py::etiquetar` | POST `{OLLAMA_BASE_URL}/api/chat` (modelo `GEN_MODEL`) | documentos ya limpios con `metadatos.source` |
| `scoring.py::puntuar` | — (solo numpy) | chunks + sus embeddings (requiere tag previo) |
| `dedup.py::deduplicar` | — (solo numpy) | chunks + embeddings (requiere scoring previo) |
| `[retrieval]` (pendiente) | `embeddear_consulta` + `obtener_cliente_chroma` + `get_collection` | `config.COLLECTION_NAME` |

Regla de dependencia por `metadatos`: **load → tag → scoring → dedup → index**. Cada módulo asume que el anterior corrió (ver §5).

---

### Detalle por archivo

### `config.py` — configuración central

Únicamente constantes. **Ningún módulo contiene parámetros mágicos**; todo ajuste se hace aquí.

| Constante | Valor | La lee | Para qué |
|---|---|---|---|
| `BASE_DIR` / `DATA_DIR` / `OUTPUT_DIR` / `LOG_DIR` | `Path` | todos | rutas estables (repositorio relativo) |
| `CHUNK_SIZE` | 800 | `chunk.py` (default), `scripts/eval_coherencia_chunks.py` | caracteres por chunk. **12%** de overlap (`CHUNK_OVERLAP=100`). Elegido ≤ contexto del modelo de embedding (all-minilm: 512 tokens ≈ 1000–1100 chars español): 800 chars cabe siempre sin truncarse. |
| `EMBED_PROVIDER` / `EMBED_MODEL` | `ollama` / `locusai/all-minilm-l6-v2` | `embed.py` | vector **384 dim** (verificado). |
| `EMBED_DIM` | 384 | nadie hoy (documental) | dimensión esperada del modelo; debe coincidir con `EMBED_MODEL` (si cambias a Gemini: 3072 y **regenerar índice**). |
| `EMBED_BATCH_SIZE` | 30 | `embed.py::embeddear` | lote `/api/embed`; 30 va bien en Ollama local. |
| `GEN_PROVIDER` / `GEN_MODEL` | `ollama` / `llama4:scout` | `tag.py` (pendiente: `generate.py`) | clasificación offline (hoy) y generación online (mañana). Modelos distintos a propósito: el tagging no necesita un 70B+ para JSON cerrado. |
| `LLM_TEMPERATURE` / `LLM_TIMEOUT` | 0.5 / 60 | pendiente (generación) | aún no usado por `tag.py` (timeout hardcodeado 120, ver §6). |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | `embed.py`, `tag.py`, `scripts/test_embeddings.py` | un solo lugar para el endpoint. |
| `CHROMA_DIR` | `output/chroma_db` | `index.py` | persistencia (gitignored, regenerable). |
| `COLLECTION_NAME` | `deporte_municipal` | `index.py` + `[retrieval]` | una colección por corpus. **Nombro definitivo: los embeddings no se pueden mezclar.** |
| `COSINE_SPACE` | `cosine` | `index.py::crear_coleccion` | coherente con vectores normalizados (verificado: ‖v‖=1). |
| `TOP_K` | 3 | `[retrieval]` | nº chunks por consulta (enunciado: documentado ✓). |
| `MAX_CHUNKS` | 500 | importado en `pipeline.py` (no aplicado aún, ver §6) | tope de chunks en el índice. |
| `LOG_FILE` / `LOG_LEVEL` | `logs/rag.log` / `INFO` | pendiente (`logging_utils.py`) | logging online del enunciado (pregunta, k, nº chunks, tiempo, modelo). |

### Módulos del pipeline (load → clean → chunk → embed → index)

#### `src/load.py` — abstracción de formato

**Lógica de diseño:** aislar el formato del pipeline. Añadir un formato nuevo = 1 loader + 1 entrada en `_LOADERS`, sin tocar `pipeline.py` (principio abierto/cerrado).

- `_LOADERS` (`L42`): registro `ext→loader` (`.pdf`, `.txt`, `.md`, `.csv`).
- `_load_pdf` (`L15`): `PyPDFLoader`. Devuelve un `Document` por página. **Nota:** LangChain pone `metadatos.source` con la ruta completa, no solo el nombre (ver §5).
- `_load_text` (`L20`): `TextLoader` utf-8 para `.txt`/`.md`.
- `_load_csv` (`L25`): **1 fila = 1 Document** → texto plano `col: valor | col: valor` + `metadatos.row`. Lógica: las filas son registros atómicos; convertir a texto plano hace el CSV compatible con el mismo chunker que el texto.
- `cargar_archivos(rutas)` (`L50`) — punto de entrada público. Acepta `str | list[str]` (archivos o carpetas, `rglob` ordenado). **Guarantiza la invariant `metadatos.source`** mediante `setdefault` después de cargar: **ningún módulo posterior puede depender de que el loader haya puesto la fuente** — este módulo lo garantiza.

#### `src/clean.py` — normalización

**Lógica de diseño:** normalizar ANTES de trocear: los PDF/CSV traen BOM, chars de control y espaciado irregular; limpiarlos evita cortes dentro de ruido y produce chunks más homogéneos (verificado: CSV escrito con BOM llega a clean con `\ufeff` y sale limpio).

- `_normalizar_page_content(texto)` (`L11`) — 5 reglas en orden: (1) 3+ saltos→2, (2) 2+ espacios/tabs→1, (3) líneas vacías repetidas, (4) espacios a ambos lados de línea, (5) chars de control + BOM.
- `limpiar(documentos | str)` (`L35`) — **doble contrato:** `str→str` (tests/scripts) y `list[Document]→list[Document]` (pipeline). Crea `Document` nuevos con copia shallow de metadatos (no muta los de entrada). *Corregido:* la versión anterior envolvía el string en lista y crasheaba en los tests.

#### `src/chunk.py` — troceado

**Lógica de diseño:** `RecursiveCharacterTextSplitter` con separadores `["\n\n", "\n", ". ", " ", ""]`: cortar **primero en límites semánticos** (párrafo → frase) y solo luego en espacios; el overlap (12%) mantiene coherencia temática entre chunks adyacentes (verificado con `scripts/eval_coherencia_chunks.py`).

- `trocear(documentos, chunk_size, chunk_overlap)` (`L12`):
  - Acepta `str` (envuelve en `Document(source="inline")`) para tests/scripts.
  - **Propaga la metadatos del documento padre** a cada chunk (`chunk_index` secuencial **por documento**). Esto es el **puente TAG→CHUNK**: `doc_category`, `tags` y `relevancia_llm` puestos por `tag.py` pasan automáticamente a los chunks sin hacer nada más.
  - `chunk_index` es 0..n **por fuente**, no global (el id global lo asigna `pipeline.py`, ver §3.2·index).

#### `src/embed.py` — vectores

**Lógica de diseño:** un único módulo conoce el endpoint de Ollama. La **condición RAG ineludible** es que el embedding del índice y el de la consulta usan el **mismo modelo**: por eso `embeddear` y `embeddear_consulta` leen `config.EMBED_MODEL` por defecto.

- `_obtener_client()` (`L12`): `requests.Session`.
- `embeddear(textos, model)` (`L17`): endpoint batch `/api/embed` por lotes de `EMBED_BATCH_SIZE` (timeout 120 s). Devuelve `list[list[float]]` 384 dims.
- `embeddear_consulta(pregunta, model)` (`L45`): embedding de **una sola** consulta (timeout 30 s). **Este es el punto de entrada de la fase online** que usará `retrieve.py` (no se invoca aún).
- `exportar_json(chunks, ruta)` (`L62`): persiste `text+metadatos+embedding` para inspección/debug (`output/embeddings.json`); se activa con `ejecutar_pipeline(..., export_embeddings=True)`.

#### `src/index.py` — persistencia ChromaDB

**Lógica de diseño:** un solo módulo toca Chroma. Cliente **persistente** (se indexa una vez para varias consultas), **regenerable** (`recreate`) y con espacio **cosine** (coherente con vectores normalizados).

- `obtener_cliente_chroma(persist_dir)` (`L12`): `PersistentClient`, `anonymized_telemetry=False`.
- `crear_coleccion(client, nombre)` (`L20`): `get_or_create_collection` con `hnsw:space=cosine`.
- `indexar(ids, embeddings, documents, metadatoss, recreate)` (`L31`):
  - Contratos: ids únicos; embedding/document/metadatos alineados por posición.
  - `_sanear(meta)` (`L60`): **la defensa única contra los tipos que Chroma rechaza** → `None`→`"null"`, listas/tuplas/sets→`str`. Esto hace que la metadatos de `scoring`/`tag` llegue tal cual.
  - `assert count == len(ids)` como verificación rápida.
  - *Nota de uso:* ejecutar dos veces **sin** `recreate` fallaría por ids duplicados; para regenerar siempre usar `recreate_index=True`.

### Módulos nuevos (tag / scoring / dedup)

#### `src/tag.py` — etiquetado semántico (Llama4)

**Lógica de diseño:** una **taxonomía cerrada** de 6 categorías (`tarifas|normativa|reservas|abonos|instalaciones|agenda`) + tags libres + `relevancia 0-1` produce un campo filtrable en Chroma (`where={"doc_category": ...}`) que sube la precisión del retrieval **por intención** y da una medida de fiabilidad del contexto. **Se etiqueta por fuente, no por documento/chunk:** un PDF es una entidad de categoría única; llamar al LLM (llama4:scout, 108B Q4 — pesado) **una vez por archivo** en vez de una vez por chunk ahorra ~90% de las llamadas y garantiza consistencia dentro del documento.

- `PROMPT` (`L14-24`): prompt de clasificación con salida tipo **esquema JSON** para el dominio seleccionado (deporte municipal) que actúa o sirve para anclar el criterio de relevancia.
- `_parse_json(texto)` (`L26`): limpia "fences" ```` ```json` / ```` ``` ```` y hace `json.loads`. **Sin fallback:** si el LLM rompe el JSON, se lanza una excepción.
- `etiquetar(documentos)` (`L33`):
  - Agrupa por `metadatos["source"]` (primer documento por fuente) y un **único POST `/api/chat` por fuente** (timeout 120 s, texto truncado a 6000 chars).
  - Escribe los metadatos: `doc_category` (str), `tags` (**str JOIN con `;`** — Chroma no filtra listas), `relevancia_llm` (tipo float).
  - **Muta la metadatos de los documentos que recibe** y devuelve la misma lista (por eso va *antes* de `trocear`: la chunking la hereda).
  - Contrato: **requiere `metadatos["source"]`** (KeyError si no existe — lo garantiza `load.py`).
  - *Nota:* no usa `config.LLM_TEMPERATURE` / `LLM_TIMEOUT` (timeout hardcodeado 120).

#### `src/scoring.py` — puntuación semántica

**Lógica de diseño:** un único float `semantic_score [0,1]` por chunk que fusiona **4 señales de naturaleza distinta**, de modo que ninguna domina (el LLM puede alucinar; los vectores a veces miden proximidad temática sin relevancia semántica):

```text
semantic_score = 0.45·relevancia_LLM     (juicio: utilidad para las preguntas del dominio)
               + 0.20·centralidad        (proximidad al centroide del corpus = representatividad del chunk)
               + 0.20·(1 − redundancia)  (no-duplicado como señal informativa)
               + 0.15·autoridad          (reglamento/normativa 1.0,
                                          precios/tarifas 0.9, agenda 0.6, CSV genérico 0.7)
```

- `AUTORIDAD` (`L12`): prioriza por **nombre de archivo** (substring): `reglamento`/`normativa`→1.0, `precios`/`tarifas`→0.9, `agenda`→0.6, default 0.7 para CSV.
- `_autoridad(source)` (`L16`).
- `puntuar(chunks, embeddings)` (`L23`): centraide del corpus + matriz `sim = E@E.T` (los vectores de all-minilm están normalizados → coseno directo); por chunk calcula centralidad, redundancia= `sim[i].max()` y escribe `metadatos["semantic_score"]` (clamped 0–1, 4 decimales).
  - **Dependencia:** asume `relevancia_llm` ya presente (si no, `get(..., 0.0)` el score cae a solo señal vectorial, no produce fallo).
  - **Orden: EMBED y TAG primeramente, DEDUP después** (dedup ordena por este campo de puntuación).

#### `src/dedup.py` — deduplicación semántica

**Lógica de diseño:** PDFs del dominio elegido (p. ej. `PreciosPublicos2026.pdf` vs `Tarifas_deportivas.pdf`) generan chunks casi idénticos **semánticamente** (no lingüísticamente). Se descarta lo que sobra, manteniendo el chunk de mayor puntuación, `semantic_score`, de cada similitud cercana. Protege `MAX_CHUNKS`, reduce ruido en retrieval y es de bajo coste por ser un método offline (una matriz coseno, sin llamada extra).

- `deduplicar(chunks, embeddings, umbral=0.93)` (`L10`):
  - `sim = E@E.T`; ordena los índices por `semantic_score` descendente.
  - Devuelve **tupla `(chunks_kept, embeddings_kept)`** — es la única función de la cadena que no cambia (a la salida de dedup va directo a `indexar`).
  - `kept.sort()` al final **restaura el orden original** para que los ids `chunk_N` de `pipeline.py` sigan siendo estables.
  - **Dependencia explícita de scoring:** si no ha corrido `scoring.puntuar`, el key es 0.0 para todos → orden estable original (funciona, pero sin criterio semántico).
  - Umbral 0.93 = punto de partida para all-minilm-l6-v2 local de Ollama.


### Orquestador - el archivo `pypeline.py`

#### `src/pipeline.py`

- `ejecutar_pipeline(rutas, chunk_size, chunk_overlap, persist_dir, collection_name, recreate_index, export_embeddings)`: encadena LOAD→CLEAN→CHUNK→EMBED→(export)→INDEX. Cada etapa loguea nº de documentos/chunks con `print` con timestamp (logging a fichero pendiente: `LOG_FILE`/`LOG_LEVEL` aún no cableados). Devuelve `dict` con métricas `{num_documents, num_chunks, embedding_dim}`.
  - **Lógica:** es el único punto de entrada offline; `main.py` (CLI pendiente) y `app.py` (Streamlit pendiente) solo lo invocan.
  - *Estado:* aún **no** incorpora TAG/SCORING/DEDUP (la integración está escrita en `integracion_retrieval.md`).

### Scripts de experimentación (para el informe)

| Script | Qué mide | Para qué sirve |
|---|---|
| `scripts/test_embeddings.py` | similitud coseno por pares de frases (3 relacionadas + 1 irrelevante) con varios modelos → **margen (relevante − irrelevante)** | justifica la elección de `EMBED_MODEL` con criterio objetivo, no a ojo. |
| `scripts/eval_coherencia_chunks.py` | similitud adyacentes vs pares aleatorios, margen entre los mismos | justifica `CHUNK_OVERLAP` (margen > 0.05 = el overlap mantiene coherencia). |

### Tests

| Test suite | Qué cubre | Notas |
|---|---|---|
| `tests/test_clean.py` | BOM, espacios, líneas vacías, chars de control, texto vacío/yá limpio; **métricas para el informe** (ratio compresión <1, palabras ≥3 chars no se pierden) | El skip espera `data/PreciosPublicos2026.pdf` para verificar números de tarifas en PDF real. |
| `tests/test_chunks.py` | tamaño, overlap presente entre adyacentes, metadatos (`source` + `chunk_index`), preservación de contenido (números/horarios/porcentajes intactos); **sweep `chunk_size` 400–1200** para el experimento del informe | |
| *(falta)* | `tag`/`scoring`/`dedup` | vectores sintéticos en scoring/dedup (offline). |

---

## Contrato de metadatos (lo que contiene Chroma ahora mismo)

| Clave | Tipo | Puesta por | Para qué (retrieval/generación) |
|---|---|---|---|
| `source` | str | `load.py` | cita de fuentes en UI y logging. ⚠️ PDF: ruta `data\x.pdf`; TXT/CSV: basename. |
| `row` | int | `load.py` (solo CSV) | localizar fila en CSV origen. |
| `chunk_index` | int | `chunk.py` (por fuente) | orden dentro del documento. |
| `doc_category` | str (6 cerradas: `tarifas·normativa·reservas·abonos·instalaciones·agenda`) | `tag.py` | `where` filtrado por intención. |
| `tags` | **str joinada `;`** | `tag.py` | display/filtrado (no usar `where` con listas). |
| `relevancia_llm` | float 0–1 | `tag.py` | entrada de `scoring`. |
| `semantic_score` | float 0–1 | `scoring.py` | **rerank + dedup**. |

Id de vectores: `chunk_0 … chunk_N` (secuencia global, la asigna `pipeline.py`). Espacio coseno · colección `deporte_municipal` · 384 dims · persistido en `output/chroma_db`.

---

## Cadenas y orden de ejecución (contrato)

```text
load.cargar_archivos        → docs con metadatos.source   (invariant)
tag.etiquetar               → + doc_category, tags, relevancia_llm   (requiere source)
chunk.trocear               → chunks que HEREDAN metadatos del tag   (puente)
embed.embeddear             → embeddings 384-dim
scoring.puntuar             → + semantic_score             (requiere relevancia_llm)
dedup.deduplicar            → (chunks, embeddings) filtrados (requiere semantic_score)
index.indexar               → ChromaDB (sanea tipos; recreate=True para regenerar)
```

---

## Problemas conocidos y notas de ajuste

1. **`(1 − redundancia)` siempre es 0.0**: los vectores están normalizados → `sim[i,i] = 1.0` → `redundancia = 1.0` para todos los chunks. Scores actuales (4 chunks sintéticos, 2 casi duplicados de piscina + 1 horario + 1 agenda):

   | chunk | score actual | redundancia (sin diagonal) |
   |---|---|---|
   | abono piscina (empadronado) | 0.7191 | 0.745 |
   | tarifa abono mensual (≈duplicado) | 0.7076 | 0.745 |
   | horario piscina | 0.5069 | 0.722 |
   | agenda baloncesto | 0.4195 | 0.575 |

   **Se solventará con (1 línea en `scoring.py::puntuar`):** `sim[i, i] = 0.0` (o `np.fill_diagonal(sim, 0)`) antes de calcular la puntuación máxima. El indicador de redundancia distinguiría los dos chunks casi duplicados.
2. **Umbral dedup 0.93:** subir a 0.95 si se deduplica demasiado demasiado; bajar a 0.90 si queda redundancia (p. ej. PreciosPublicos2026 vs Tarifas_deportivas).
> Esto tengo que verlo bien, que estoy pensando otras cosas.
3. **`tag.py` asume `metadatos["source"]`** (KeyError) y timeout hardcodeado (120 s). Mejora pendiente: try/except en `_parse_json` con default `doc_category="instalaciones"`.
4. **Orden forzado:** tag→scoring→dedup como en §5; `deduplicar` sin `puntuar` degrada a orden estable (funciona, sin criterio semántico).
5. **`pipeline.py` no ejecuta aún tag/scoring/dedup** (la integración está escrita en `src/integracion_retrieval.md`) — es la primera tarea de cabecera para que el índice lleve la metadatos completa.


