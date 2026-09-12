# RAG Deporte Municipal

Este proyecto es un sistema **RAG (Retrieval Augmented Generation)** para responder preguntas sobre **deporte municipal de Madrid**: tarifas, abonos, reservas, horarios e instalaciones.

Carga un corpus real (PDF + CSV + TXT), lo trocea, genera embeddings, mejora la calidad con etiquetado LLM + scoring + deduplicación, y lo guarda en **ChromaDB** para retrieval semántico (estas últimas son opionales).

## Qué hace

Pipeline offline completo:

```text
LOAD → CLEAN → TAG (LLM) → CHUNK → EMBED → SCORING → DEDUP → INDEX
```

- **Corpus propio** en `data/`: 9 PDF normativos/tarifas, 7 CSV de datos abiertos y 1 TXT de agenda.
- **Multi-proveedor** sin tocar código: cambios de proveedor por variable de entorno (`ollama`, `huggingface` o `google`) en embeddings, tagging y generación.
- **Bloque TSD (Tag–Scoring–Dedup)**: las etiquetas del LLM viajan con cada chunk, se puntúa semánticamente y se descartan los chunks casi duplicados. El índice entra más denso en señal y menos en ruido.
- **Índice regenerable** en ChromaDB, persistente y con verificación de inserción.
- **Logging por consola** de cada fase + dict de métricas para el informe.

Preguntas de ejemplo:

- ¿Cuánto cuesta el abono de piscina?
- ¿Hay descuentos para menores, mayores, estudiantes o personas en paro?
- ¿Puedo reservar siendo no empadronado?
- ¿Qué pasa si cancelo una reserva con coste?

> Hay un documento en [queries/preguntas.json](./queries/preguntas.json) con todas las preguntas generadas para pruebas de evaluación.

## Estructura del proyecto

```text
project-rag-hmd-deporte-municipal/
├── README.md
├── requirements.txt
├── .env.example
├── config.py                # todas las variables (leídas de .env)
├── src/
│   ├── pipeline.py          # orquestador: encadena el pipeline completo
│   ├── load.py              # LOAD: PDF/TXT/MD/CSV → Documents (una fila CSV = un doc)
│   ├── clean.py             # CLEAN: normaliza saltos, espacios, BOM, control chars
│   ├── chunk.py             # CHUNK: RecursiveCharacterTextSplitter (Level 2)
│   ├── embed.py             # EMBED: multi-proveedor (ollama / huggingface / google)
│   ├── index.py             # INDEX: ChromaDB persistente, métrica coseno
│   └── tsd/
│       ├── tag.py           # TAG: etiquetado LLM por fuente (taxonomía cerrada)
│       ├── scoring.py       # SCORING: relevancia + centralidad + redundancia + autoridad
│       └── dedup.py         # DEDUP: greedy por similitud coseno
├── data/                    # corpus (9 PDF + 7 CSV + 1 TXT)
├── queries/                 # preguntas de evaluación
├── scripts/                 # validación de chunks y embeddings
├── tests/                   # pruebas pytest de load/clean/chunk
└── output/                  # índice Chroma y embeddings exportados (gitignored)
```

## Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/<tu-usuario>/project-rag-hmd-deporte-municipal.git
cd project-rag-hmd-deporte-municipal
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

### 3. Configura `.env`

Copia el ejemplo y edita los valores:

```bash
cp .env.example .env
```

Elige proveedor en cada sección:

| Variable | Valores | Default |
|---|---|---|
| `EMBED_PROVIDER` / `TAG_PROVIDER` / `GEN_PROVIDER` | `ollama` / `huggingface` / `google` | `huggingface` |

Si eliges `huggingface`, configura `HF_DEVICE` (`cpu` o `cuda`) y, solo si el modelo es gated, pon tu token en `HF_TOKEN`.
Si eliges `google`, pon tu clave en `GOOGLE_API_KEY`.
Si eliges `ollama`, sube Ollama con el modelo de embeddings y el LLM descargados (`OLLAMA_BASE_URL` por defecto: `http://127.0.0.1:11434`).

**Ojo:** si cambias el modelo de embeddings o el chunking, **regenera el índice** (ver [Uso](#uso) → `--recreate-index`).

## Uso

### Indexar el corpus

Desde la raíz del proyecto:

```bash
python -m src.pipeline data --recreate-index
```

La consola va mostrando el avance fase a fase:

```text
2025-09-01 12:00:00 [LOAD] 234 documentos cargados
2025-09-01 12:00:01 [CLEAN] 234 documentos normalizados
2025-09-01 12:00:02 [TAG] 15 fuentes etiquetadas en 43.2s (proveedor: huggingface, modelo: ...)
2025-09-01 12:00:03 [CHUNK] 1289 chunks | min 40 · p25 612 · media 745 · p75 998 · max 1000 · cortos(<50) 12
2025-09-01 12:00:05 [EMBED] 1289 vectores de 1024 dims
2025-09-01 12:00:06 [SCORE] 1289 chunks | mín 0.210 · media 0.612 · máx 0.914
2025-09-01 12:00:06 [DEDUP] umbral 0.93 -> descarta 302 de 1289 (23.4%)
2025-09-01 12:00:07 [INDEX] colección 'deporte_municipal': 987 vectores (987 en esta inserción)
2025-09-01 12:00:07 [FIN] pipeline completado en 61.3s
```

`ejecutar_pipeline` devuelve además un dict de métricas (`num_documentos`, `num_chunks_pre_dedup`, `num_chunks_post_dedup`, `chunks_descartados`, `dim_embedding`, `chunk_stats`, `tiempo_total_s`) listo para el informe.

### Parámetros opcionales del pipeline

```powershell
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

**Ejemplo de pipeline completo:**

```powershell
python -m src.pipeline data --recreate-index
```


### Validación (scripts y tests)

```bash
# Coherencia de los cortes: similitud coseno entre chunks adyacentes vs. aleatorios
python -m scripts.eval_coherencia_chunks

# Pruebas de load/clean/chunk
python -m pytest tests/ -v
```

## Configuración (.env) sin tocar archivos de código

Admite los siguientes:
* Tres providers distintos (ollama [local] / google / huggingface)
* Size de los chunks y overlaping.
* Tamaño del lote de embeddings.
* Nombrado de la colección de vectores
* (opcional) Exportación de embeddings a JSON.
* (opcional) tag, scoring y dedupicación de embeddings con similitud semántica.
* (opcional) umbral de deduplicación semántica, unido al anterior.

Todo lo configurable vive en `config.py`, que le `.env` con defaults. Para cambiar algo, edita `.env`:

| Variable | Default | Qué controla |
|---|---|---|
| `EMBED_PROVIDER` | `huggingface` | Proveedor de embeddings |
| `TAG_PROVIDER` | (sigue a `GEN_PROVIDER`) | Proveedor del etiquetado LLM |
| `GEN_PROVIDER` | `huggingface` | Proveedor de generación |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Endpoint de Ollama |
| `OLLAMA_EMBED_MODEL` / `OLLAMA_GEN_MODEL` / `OLLAMA_TAG_MODEL` | (ver `.env.example`) | Modelos Ollama |
| `HF_EMBED_MODEL` / `HF_GEN_MODEL` | (ver `.env.example`) | Modelos HuggingFace |
| `HF_DEVICE` | `cuda` | `cpu` / `cuda` (cuda exige GPU NVIDIA) |
| `HF_TOKEN` | — | Solo modelos gated |
| `GOOGLE_API_KEY` | — | Necesaria si usas `google` |
| `GOOGLE_EMBED_MODEL` / `GOOGLE_GEN_MODEL` | (ver `.env.example`) | Modelos Gemini |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `100` | Parámetros del troceado |
| `EMBED_DIM` | `1024` | Dimensión declarada del embedding (el pipeline avisa si no coincide con la real) |
| `EMBED_BATCH_SIZE` | `100` | Tamaño de lote de embedding |
| `EXPORT_EMBEDDINGS` | `true` | Exporta `output/embeddings.json` al terminar |
| `TAG_SCORING_DEDUP` | `true` | Activa/desactiva el bloque TSD completo |
| `DEDUP_UMBRAL` | `0.93` | Umbral coseno de dedup |
| `CHROMA_DIR` | `./output/chroma` | Directorio del índice |
| `COLLECTION_NAME` | `deporte_municipal` | Nombre de la colección |

**IMPORTANTE:** si cambias `EMBED_*`, `CHUNK_*` o el corpus, ejecuta de nuevo el pipeline con `--recreate-index`. Los vectores son función directa del texto troceado; un índice no regenerado produce resultados inválidos.

## Cómo funciona cada fase

| Fase | Módulo | Qué hace |
|---|---|---|
| LOAD | `src/load.py` | Carga PDF (PyPDF), TXT/MD (Text) y CSV. Cada **fila** de CSV se convierte en un documento legible `campo: valor \| campo: valor`. Detecta encoding (utf-8-sig → cp1252 → latin-1). |
| CLEAN | `src/clean.py` | Normaliza antes de trocear: saltos múltiples, espacios extra, BOM y caracteres de control. No toca metadatos. |
| TAG | `src/tsd/tag.py` | Pregunta al LLM **una vez por fuente** (no por página) y propaga `doc_category`, `tags` y `relevancia_llm` a todos los documentos. Taxonomía cerrada: `tarifas`, `normativa`, `reservas`, `abonos`, `instalaciones`, `agenda`. Si el LLM devuelve JSON malformado, usa valores por defecto en vez de romper el pipeline. |
| CHUNK | `src/chunk.py` | `RecursiveCharacterTextSplitter` con separadores semánticos (párrafo → frase → espacio → letra). Cada chunk hereda la metadata del documento padre. |
| EMBED | `src/embed.py` | Genera vectores por lotes con el proveedor elegido. Embeddings normalizados (coherentes con la métrica coseno de Chroma). |
| SCORING | `src/tsd/scoring.py` | `semantic_score = 0.45·relevancia_LLM + 0.20·centralidad + 0.20·(1 − redundancia) + 0.15·autoridad`. Redundancia y dedup con FAISS (memoria O(n·d), no O(n²)). |
| DEDUP | `src/tsd/dedup.py` | Recorre chunks por score descendente y descarta los que superan `DEDUP_UMBRAL` de similitud con uno ya conservado. Implementación incremental con FAISS (sin matriz completa). |
| INDEX | `src/index.py` | Inserta en ChromaDB por lotes, sanea metadatos (Chroma no acepta `None` ni listas) y verifica que todos los ids quedaron vivos. |

La dependencia entre fases va por **metadata**: `tag` escribe → `chunk` propaga → `scoring` lee y escribe → `dedup` lee → `index` sanea.

## Corpus y fuentes

Corpus descargado de la sede de **datos abiertos del Ayuntamiento de Madrid** (fecha de descarga: 2026). Datos públicos, uso educativo, sin secretos.

| Archivo | Origen |
|---|---|
| `PreciosPublicos2026.pdf` | [Precios públicos centros deportivos 2026](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/ficheros/TarifasD/PreciosPublicos2026.pdf) |
| `Tarifas_deportivas.pdf` | [Tarifas de servicios en centros deportivos](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/ficheros/TarifasD/Tarifas_deportivas.pdf) |
| `eli-es-md-01860896-reg-2012-10-15-(1)-dof-spa.pdf` | [Reglamento de instalaciones deportivas](https://sede.madrid.es/eli/es-md-01860896/reg/2012/10/15/(1)/dof/spa/pdf) |
| `PiscinasAireLibre2026.pdf` | [Piscinas de verano 2026 aire libre](https://www.madrid.es/UnidadesDescentralizadas/Deportes/EspecialInformativo/Verano2026/ficheros/PiscinasAireLibre2026.pdf) |
| `DecretoAnulacionReservasConCoste.pdf` | [Decreto anulación de reservas con coste](https://www.madrid.es/UnidadesDescentralizadas/Deportes/ContenidoGenerico/ContenidoGenerico2024/Ficheros/DecretoAnulacionReservasConCoste.pdf) |
| `BasesReguladoras47JDM_1Sp.pdf` / `BasesDeportesEquipo47jdm_1Sp.pdf` | [Normativa Juegos Deportivos Municipales 47](https://www.madrid.es/UnidadesDescentralizadas/Deportes/EspecialInformativo/46%20Juegos%20Deportivos%20Municipales%2025-26/normativas/normativaGeneral46jdm.pdf) |
| `20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf` | [Infografía Abono Deporte Madrid](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Faq/ficheros/infografias%20faq%202025/20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf) |
| `200186-0-polideportivos.csv` | [Polideportivos](https://datos.madrid.es/dataset/200186-0-polideportivos) |
| `200215-0-instalaciones-deportivas.csv` | [Instalaciones deportivas básicas](https://datos.madrid.es/dataset/200215-0-instalaciones-deportivas) |
| `210227-0-piscinas-publicas.csv` | [Piscinas públicas](https://datos.madrid.es/dataset/210227-0-piscinas-publicas) |
| `300390-0-areas-deportivas.csv` | [Áreas de actividades deportivas](https://datos.madrid.es/dataset/300390-0-areas-deportivas) |
| `212504-0-agenda-actividades-deportes.csv` | [Agenda de actividades deportivas](https://datos.madrid.es/dataset/212504-0-agenda-actividades-deportes) |
| `300085-0-deportes_abonos.csv` | [Abonados en centros deportivos](https://datos.madrid.es/dataset/300085-0-deportes_abonos) |
| `300097-0-deportes-descuentos.csv` | [Descuentos en instalaciones](https://datos.madrid.es/dataset/300097-0-deportes-descuentos) |
| `211549-0-juegos-deportivos-actual.txt` | [Juegos deportivos municipales vigente](https://datos.madrid.es/dataset/211549-0-juegos-deportivos-actual) |

Los CSV los puedes regenerar descargando directamente cualquier dataset de [datos.madrid.es · grupo deporte](https://datos.madrid.es/group/deporte). El loader los recargará igual aunque cambien de esquema ligero.

## Documentación técnica

- [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md) — estado completo del pipeline, módulos, flujo y métricas.
- [`docs/ANEXO_CHUNKING.md`](docs/ANEXO_CHUNKING.md) — cómo cambiar el nivel de chunking y la validación obligatoria.

## Roadmap

- [x] Corpus multi-formato (PDF + CSV + TXT)
- [x] Pipeline offline completo con bloque TSD
- [x] Multi-proveedor (Ollama / HuggingFace / Google)
- [x] Índice Chroma verificado y regenerable
- [ ] Fase online (retrieval): `--query` con filtro por `doc_category` y rerank por `semantic_score`
- [ ] Generación RAG con abstención: `--ask`
- [ ] UI Streamlit: chat + chunks visibles + tabla de métricas

## Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| El pipeline avisa `AVISO: dim X != EMBED_DIM` | El modelo descargado no coincide con `EMBED_DIM` en `.env` | Poner `EMBED_DIM` a la dimensión real del modelo (p. ej. `384` para `all-MiniLM-L6-v2`, `1024` para `Qwen3-Embedding-0.6B`) **y** regenera el índice con `--recreate-index`. |
| Ollama responde sin campo `embeddings` | Versión antigua | Update Ollama a >= 0.9 (el batch `/api/embed` lo exige) o usa `EMBED_PROVIDER=huggingface`. |
| Chroma `greater than max batch size` | Indexado sin lotes | El código ya inserta por lotes (`_BATCH_ADD = 2000`) si no aparece, tu colección está dañada: borra `output/` y regenera. |
| El TAG usa la categoría por defecto | El LLM devolvió JSON malformado | Revisa que el modelo de `TAG_PROVIDER` sea capaz de emitir JSON. Para el pipeline no es un error: se marca en la consola y continúa. |
| `TSD scoring + dedup` "se va" sin acabar (sin más log) | OOM: la matriz de similitud n×n no cabe en RAM (n≈130k → ~68 GB). **Resuelto en el código** (scoring con FAISS k=2, dedup incremental; memoria O(n·d)). Verificar el coste real con `python -m scripts.benchmark_tsd`. |
| La dedup descarta casi todo el corpus (p. ej. >90 % con `DEDUP_UMBRAL=0.93`) | El umbral es ajustable: **bájalo** (p. ej. `0.85`) para conservar más chunks, **súbelo** (p. ej. `0.95`) para dedup más. Con chunking por fuente (CSV), fuentes repetitivas se dedupan a masas. |
| Cambié chunking/modelo y el retrieval falla | Índice no regenerado | Ejecuta `python -m src.pipeline data --recreate-index`. |
