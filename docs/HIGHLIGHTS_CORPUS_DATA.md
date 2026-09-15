# Highlights de `feature/corpus-data`

**Objetivo:** documentar qué aporta este bloque de trabajo para la fase de corpus e indexación offline (Parte 1 y Parte 2 de la rama `feature/corpus-datos`).

La línea base que fija el enunciado es: *cargar corpus (>=2 formatos) → limpiar → trocear → embeddings → ChromaDB con `source` → indexado separado de la consulta → logging básico → recreate del índice*. Lo que viene a continuación marca, en cada punto, **lo mínimo exigido** y **lo que hemos superado**.

---

## 1. El corpus (Parte 1)

| Dimensión | Exigido | Aportado en `feature/corpus-data` |
|---|---|---|
| Nº de formatos distintos | >=2 | **3 presentes** en `data/` (8 PDF + 7 CSV + 1 TXT); el loader también soporta MD (markdown) |
| Mezcla texto + estructurado | "preferible" | Sí: PDF normativos/tarifas (texto) combinados con CSV de open data estructurado + agenda TXT |
| Documentación de fuentes | enlaces + fecha | corpus real descargado de datos abertomadrid.es (polideportivos, instalaciones, piscinas, abonos, descuentos, tarifas, reglamento, decretos de reserva) |

**Por encima:**
- El CSV ya no se traga como "una fila = un documento plano": antes de trocear hay una capa de **clasificación diferencial** (`src/csv_advisor.py` y `src/csv_transform.py`) que decide el tratamiento de cada fuente: `entity_doc` (fila = entidad con ID estable → dedup `exact_key`), `grouped_doc` (tabla de hechos repetitiva → dedup `group_only`) o `row_as_doc` (fallback). La metadata (`dedup_policy`, `entity_key`, `group_key`, `chunking_hint`) viaja con el chunk hasta el índice y **la dedup de CSVs nunca es semántica** (evita perder información de entidades casi idénticas), sino por clave exacta.
- La carga soporta **archivo o carpeta recursiva** con registro de extensión (`_LOADERS`): añadir un formato nuevo = añadir una fila al dict, sin tocar el resto.
- `metadata.source` **siempre** poblado, aunque el loader no lo hiciera (default en load.py).

---

## 2. Preflight: validación preventiva antes de empezar

Una **fase preflight** es un mecanismo de validación preventiva diseñado para bloquear errores, inseguridades o configuraciones incorrectas **antes de que afecten la ejecución principal**. El término se usa en Python al menos en tres contextos, todos con la misma idea de "puerta (mecanismo) de control" (*preflight-gate*) previo:

1. **Seguridad y carga de plugins**: mecanismos implementados por librerías como *preflight-gate* validan el manifiesto de un plugin (seguridad, permisos, origen) antes de importar su código: una puerta basada en
  archivos inertes (JSON) previa a la importación real, para que nada malicioso ni no autorizado entre en el proceso. 
  > En este documento, `preflight-gate` es una referencia conceptual; no es una dependencia utilizada por este proyecto.
3. **Despliegue y calidad de código**: algunas herramientas CLI escanean la base antes de enviarla a producción (por ejemplo, variables de entorno erróneas, claves API filtradas, `print`/`console.log`
  de depuración o controles de seguridad ausentes): "listo para volar" antes del deploy, como uso práctico y dependiente de la herramienta.
3. **Inicialización del entorno**: preinicialización del intérprete (`PyPreConfig`, `Py_InitializeFromConfig`): memoria, codificaciones y configuración fijadas antes de que el núcleo cargue por completo. Se ve en el arranque de CPython (la C API `Py_InitializeFromConfig` y el struct `PyPreConfig` en `Python/pylifecycle*.c`, poblado desde las flags `-X`/`-I` y visible en runtime vía `sys.flags`); como los anteriores, es una referencia conceptual del *preflight-gate*, no una dependencia de este proyecto.

Este proyecto **aplica ese mismo espíritu a la indexación**: el pipeline empieza ahora por una fase `PREFLIGHT` **antes** de `LOAD`, que garantiza que `EMBED_MODEL` está disponible en `EMBED_PROVIDER` (Ollama `/api/tags` · HuggingFace `repo_exists` · Google listado de modelos) y **aborta** (`RuntimeError`) si no existe. Si la comprobación online no es posible (red cortada, Ollama apagado, sin API key), se degrada con aviso y hace *fallback* al caché offline de `.env`(`EMBED_DIM_MAX_*`, `verificado_por: cache_env`) en vez de romper el pipeline. El resultado (online/cache, dim máxima declarada, aviso) se refleja en la consola y en la sección 2 del informe.

---

## 3. El pipeline offline (Parte 2) — más allá de load→clean→chunk→embed→index

Se ha ido por encima de lo que es *la cadena básica* y se inserta un **bloque TSD** (Tagging–Scoring–Deduplication), porque mejora la calidad del retrieval antes de que los chunks entren al índice, además de un informe tras la indexación que resume la parametrización y refleja métricas para decisión:

```text
PREFLIGHT → LOAD → CLEAN → TAG(LLM) → CHUNK → EMBED → SCORING → DEDUP → INDEX → INFORME
  (extra)                                               (extra)                  (extra)
```

- **TAG** (`tsd/tag.py`): etiquetado semántico con LLM usando una taxonomía cerrada (`doc_category`, `tags`, `relevancia`). Etiqueta **una vez por fuente** (1 llamada LLM por documento, no por página) y propaga el resultado a todos sus chunks, luego barato y consistente.
- **SCORING** (`tsd/scoring.py`): `semantic_score` por chunk que combina relevancia del LLM, **centralidad** (coseno contra el centroide), **redundancia** (coseno máxima con el resto, vía FAISS: memoria O(n·d) en vez de O(n²)) y **autoridad** de la fuente; vuelca sus métricas por parámetro `info`.
- **DEDUP** (`tsd/dedup.py`): deduplicación **consciente de política** — chunks con `dedup_policy` exacta (`exact_key`/`group_only`/`exact_only`) se deduplica por clave y **nunca por coseno**; los
  demás, deduplicación greedy por `semantic_score` con umbral configurable (FAISS incremental). 
- **INFORME**: al terminar, `ejecutar_pipeline` ensambla el `dict` de métricas y genera el informe (ver punto 4); se puede saltar con `informe=False` (tests).

**Por encima:** todo el bloque es optativo por switch (`TAG_SCORING_DEDUP`) y el pipeline devuelve un `dict` mucho más rico que el mínimo: *tiempos por fase*, *resúmenes de cada fase*, *métricas de*
*scoring/dedup*, *dimensionalidad* y *preflight*.

---

## 4. Informe de indexación markdown (`src/informe.py`)

Como extra tras la ingesta: `output/informe_indexacion.md` se genera **automáticamente al terminar** con todo lo necesario para decidir qué parámetros tocar:

- **Parámetros aplicados** (proveedores, modelos, `EMBED_DIM` efectivo, chunking, TSD, umbral, colección, recreate…): qué ejecución fue la que dejó ese índice.
- **Preflight**: cómo se verificó el modelo (`online` o `cache_env`), dim máxima declarada, aviso.
- **Métricas por fase** (`PREFLIGHT` → `FIN`) con tiempos y resumen de cada una.
- **Chunking** (min/p25/media/p75/max, chunks < 50 = ruido probable a investigar) y **índice final** (vectores insertados vs. totales de la colección, dim, recreate).
- **TSD**: scoring (min/media/máx, % chunks ≥ 0.6, centralidad y redundancia medias) y deduplicación (antes/después, descarte exacto vs. semántico, % total).
- **Señales y criterios de decisión**: heurísticas automáticas sobre las métricas (dedup > 50 %, dedup a 0, dim desajustada, caché offline de preflight, < 50 % de chunks buenos…), con la acción
  concreta que recomiendan (bajar/subir `DEDUP_UMBRAL`, tocar `EMBED_DIM`, revisar loaders…).

El módulo **solo renderiza** (no lee config ni hace side effects), así que su salida se testea aislada, y el `e2e` corre con un proveedor fake (sin red ni Ollama). Concretamente:

- **"Se testea aislada"**: el informe es una función pura —entrada: el `dict` de métricas que devuelve `ejecutar_pipeline`; salida: el string markdown—, así que los tests fabrican ese
  `dict` a mano y solo verifican el texto generado (secciones, valores y señales presentes); no hay Chroma, `.env`, ni red que preparar.
- **"E2E con proveedor fake"**: el test de extremo a extremo del pipeline usa un proveedor de embeddings stub (devuelve vectores deterministas de dimensión fija, sin llamar a Ollama ni a
  ninguna API) y escribe el índice en un directorio temporal: el flujo completo PREFLIGHT → … → INFORME se valida sin red ni Ollama instalado.

---

## 5. Dimensión garantizada (`dim = min(dim modelo, EMBED_DIM)`)

Aquí se puede asumir erróneamente que la dim del modelo y `EMBED_DIM` coinciden. Aquí la dim del índice es **siempre** el mínimo que maneja el modelo, con una doble protección:

- **Pista al proveedor** en servidor: Ollama (`dimensions` en el payload) y HuggingFace (`truncate_dim`, *slicing Matryoshka*); Google lo acepta y lo ignora de forma explícita.
- **Garantía real en cliente** (`_corte_dim` en `src/embed.py`): `embeddear()` es la **puerta única** por la que pasan índice y consulta futura, y recorta el prefijo + renormaliza si el proveedor devuelve más de `EMBED_DIM`. La métrica coseno se mantiene. 

> *slicing Matryoshka*: técnica de *Matryoshka Representation Learning* — el modelo se entrena para que cualquier prefijo del vector siga siendo un embedding válido—; por eso recortar a las primeras `EMBED_DIM` dimensiones y renormalizar no degrada la similitud coseno. Revisar función en `src/embed.py`: `_corte_dim()` la cual recorta el prefijo y renormaliza el > vector, y `embeddear()` lo aplica como puerta única de índice y consulta.

Conversión del chequeo en niveles: `EMBED_DIM < dim modelo` → **INFO** (se indexa con `EMBED_DIM`); `EMBED_DIM > dim modelo` → **AVISO** y ajuste en memoria a la dim real del modelo (la de
`EMBED_DIM_MAX_*` declarada en el preflight). Se evita así que "el índice se crea con una dim que el modelo no soporta": **solucionado de raíz**.

---

## 6. Índice ChromaDB (`src/index.py`)

| Exigido | Aportado |
|---|---|
| Chroma persistente con `source` | Sí, **y saneado de metadata** (Chroma no acepta `None` ni listas: se normalizan) |
| `recreate`/regenerar índice | `recreate=True` con borrado seguro de la colección si no existe |
| Verificación simple | Verificación de inserción: los ids insertados deben existir en la colección |

**Como extra:** `indexar()` ahora retorna `(vivos, totales)` — la inserción verificada (ids vivos) **y** el recuento total de la colección — para que el informe muestre ambas cifras y se detecte un re-index parcial. El cliente y la creación de colección son funciones reutilables (`obtener_cliente_chroma`, `crear_coleccion`) pensadas para que la fase de retrieval (`embeddear_consulta` + filtro por `doc_category` + rerank por `semantic_score`) sea posible sin tocar el indexado.

---

## 7. Embeddings multi-proveedor (`src/embed.py`)

Para el apartado "API de LLM y embeddings" se plantean tres proveedores distintos:

- **3 proveedores intercambiables por variable de entorno** (`EMBED_PROVIDER`): `ollama` (batch `/api/embed`), `huggingface` (`SentenceTransformer`, carga perezosa + caché), `google` (REST `batchEmbedContents`). Añadir un proveedor = +1 función + 1 línea en el dict `_PROVIDERS`.
- **Coherencia de métrica**: embeddings normalizados + colección coseno → la similitud es coseno real en toda la cadena (embed, scoring, dedup, retrieval).
- **Cap de dimensión por proveedor** (punto 5) y **preflight de disponibilidad** (punto 2), con caché offline vía `EMBED_DIM_MAX_OLLAMA` / `_HF` / `_GOOGLE`: el preflight declara la dim máxima que
  maneja el modelo incluso sin red ni API key.
- **Export a JSON** (`embeddear` + `exportar_json`): volca `text+metadata+embedding` a `output/embeddings.json` para inspección/debug → ayuda al experimento de chunking del informe.
- `embeddear_consulta(pregunta)` ya lista para la fase online (condición RAG: mismo modelo que al indexar), con la misma garantía de dim que el índice.

---

## 8. Robustez y correcciones aplicadas sobre el código de esta rama

Correcciones que hacen que el pipeline **funcione de punta a punta** (evita el fallo en la fase INDEX, la pérdida de datos o el arranque contra un proveedor no soportado o inexistente):

- **IDs de chunk globales y únicos** (pipeline.py): se evita que `source::chunk_index` y `chunk_index` sea secuencial *por documento*, dado que un CSV con la misma fuente podría 
  generar ids duplicados, dado que ChromaDB sobreescribía filas. Aquí el id es único (índice global) y ninguna fila se pierde.
- **Preflight con borrado a tiempo**: un modelo de embeddings mal escrito o no descargado no rompe la ejecución en `EMBED` tras minutos de LOAD/TAG: el pipeline aborta en segundos con el mensaje: `PREFLIGHT: el modelo 'X' NO está disponible en <proveedor>`.
- **Keyword `metadatas=`** correcto en `collection.add()` (index.py) y firma de `indexar()` consistente con el *caller* de añadir/indexar (pipeline.py).
- **Fix de `metadata.get["source"]`** (tag.py): un tipo podía romper la propagación de etiquetas a todas las páginas/filas. Esto está solucionado para que no ocurra.
- **Guardas defensivos**: centroide degenerado en scoring (norma 0), `dedup` sobre lista vacía, `chunk_overlap=0` no se traga como false, error explícito si el proveedor no devuelve el campo esperado. Ellos son parte de un *must have* en tiempo de ejecución.
- **Dedup por política exacta para CSVs** (dedup.tsd + csv_advisor): la `dedup` semántica por coseno ya no descarta entidades casi idénticas (dos piscinas del mismo barrio no son duplicados semánticos
  que apenas difieren el unos pocos datos en CSV); solo computan los chunks no estructurados. 
- **Logging ASCII-safe** en consola (evita caracteres que rompen la terminal en Windows) y verificación real de inserción en lugar de un `count()` acumulado que podría no decir verdad al re-indexar.

---

## 9. Chunking con criterio (Level 2) y validación

- `RecursiveCharacterTextSplitter` con separadores **semánticos** (párrafo → frase → espacio → letra), el "multiusos" recomendado como punto de partida.
- `chunk.py` es el **puente TAG→CHUNK**: hereda la metadata del documento padre (`doc_category`, `tags`, `relevancia_llm`, y ahora también `dedup_policy`/`entity_key`/`group_key` de la capa CSV) a cada chunk → viaja gratis/barato en tokens hasta el índice. Además **respeta `chunking_hint`** del advisor (`no_chunk`/`light_chunk`): una fila de entidad de 200 caracteres no pasa por la misma cuchilla que un reglamento de 80 páginas, como regla lógica plausible.
- **Validación de coherencia** (`scripts/eval_coherencia_chunks.py`): mide similitud coseno entre chunks adyacentes vs. aleatorios para confirmar que el overlap mantiene coherencia temática ("Chunking Commandment": el dato se recupere con valor).
- Tests offline de troceado (tests/test_chunks.py) que cubren tamaño, overlap, metadata y preservación de contenido (precios/horarios no se rompen), más métricas para el experimento de `CHUNK_SIZE`/`CHUNK_OVERLAP` del informe.

---

## 10. Configuración centralizada (`config.py`)

Todo el interruptor de proveedores, modelos, `EMBED_DIM`/`EMBED_BATCH_SIZE`, `CHUNK_SIZE`/`CHUNK_OVERLAP`, `DEDUP_UMBRAL`, `CHROMA_DIR`/`COLLECTION_NAME` **y el caché offline del preflight** `EMBED_DIM_MAX_OLLAMA`
/`_HF`/`_GOOGLE` vive en `.env` vía `config.py`. Cambio de proveedor implica cambio de variable, **sin tocar código**. Tres interruptores independientes por proveedor (EMBED / GEN / TAG) con fallback: si `TAG_*` no se define, hereda desde `GEN_*`.

---

## Resumir en una frase

> La fase base (corpus multi-formato + pipeline load→clean→chunk→embed→index con Chroma) está **completa y funcional**; el plus es la **fase PREFLIGHT** (validación preventiva de disponibilidad del modelo antes de arrancar, con caché offline), la **dimensión garantizada** `min(dim modelo, EMBED_DIM)`, el **tratamiento diferencial de CSVs** (clasificación + dedup po clave, nunca semántica), el **bloque TSD** (entrega de señal limpia al índice), **el informe de indexación con señales para decidir** y la abstracción multi-proveedor con robustez suficiente para que el índice no pierda datos ni rompa contra inputs anómalos.

---

### Pendiente/externo a esta rama (para la siguiente)
- Fase online: `retrieve` (filtro por `doc_category` + rerank por `semantic_score`) y `--query`.
- La parte de generación/eval y Streamlit es `feature/generacion-eval` / `feature/streamlit-ui`.
