# HT-OUTPUT-01 — Guía técnica: outputs del pipeline (informe de indexación)

## 1. Objeto

Este documento describe los outputs que genera `ejecutar_pipeline(...)` en `output/` y, sobre todo, la **estructura y la interpretación del informe de indexación** (`output/informe_index_aaaaMMdd_hhmm.md`): qué se puede esperar en cada sección, qué reglas gobiernan la tabla de parámetros y cómo leer las señales automáticas para decidir qué ajustar.

## 2. Alcance y referencias

* **Aplicabilidad:** `src/pipeline.py` (ensambla el dict de métricas), `src/informe.py` (renderiza el markdown), `src/index.py` (datos de la sección 5).
* **Generación:** cada ejecución del pipeline escribe `output/informe_index_aaaaMMdd_hhmm.md` por defecto (fecha y hora locales de la ejecución; salvo `informe=False` en los tests). No hay flag de consola: se genera siempre al terminar.
* **Referencias:** sección *Uso* de `README.md` y tabla de `.env` (variables que aparecen en el informe).
* **Validación:** `tests/test_informe.py` (renderizado unitario, offline) y `tests/test_informe_e2e.py` (humo extremo a extremo con proveedor fake).

## 3. `output/informe_index_aaaaMMdd_hhmm.md`

Objetivo del informe: responder en una sola lectura **qué parámetros determinaron esta ejecución** y **qué tan buena quedó la indexación**, para tomar decisiones (bajar/ subir umbral, cambiar modelo, regenerar).

### 3.1 Estructura (7 secciones, siempre en el mismo orden)

| Sección | Contenido | Para qué sirve |
|---|---|---|
| 1. Parámetros aplicados | Tabla `variable · valor · nota` con los parámetros que influyeron en esta ejecución | Verificar la configuración efectiva (incluye overrides por flag, p. ej. `--chunk-size 500` aparece como 500 aunque `CHUNK_SIZE` en `.env` sea otro) |
| 2. Preflight | `verificado_por` (`online` / `cache_env`), dim máxima declarada (si la hay), aviso | Detectar ejecuciones degradadas (red cortada / sin API key) antes de confiar el dim |
| 3. Métricas por fase | Tiempo (s) y resumen de cada fase de `PREFLIGHT` a `FIN` | Localizar la fase lenta o dónde se descartaron chunks |
| 4. Chunking | min/p25/media/p50/p75/max de longitud y nº de chunks cortos (< 50) | Evaluar si el troceado es razonable para el corpus |
| 5. Índice final (ChromaDB) | Colección, space, dim, vectores insertados vs. totales, si se recreó | Confirmar que la inserción quedó verificada y el tamaño esperado |
| 6. TSD (scoring + dedup) | Distribución de `semantic_score` + desglose de descartados (exactos vs. semánticos) + **cobertura por categoría** (pre/post dedup) y top 3 tags | Medir densidad de señal del índice, efecto de `DEDUP_UMBRAL` y qué intenciones quedan cubiertas |
| 7. Señales y criterios de decisión | Heurísticas automáticas interpretadas (ver 3.6) | Qué revisar o ajustar a partir de esta ejecución |

Encabezado común a todas las secciones: fecha de ejecución (UTC local), rutas del corpus y tiempo total.

### 3.2 Sección 1 — Parámetros aplicados

Reglas de la tabla (implementadas en `src/pipeline.py`):

* **Siempre se muestran** (son irrelevantes al proveedor y afectan a esta ejecución):
  `EMBED_PROVIDER`, `EMBED_MODEL`, `EMBED_DIM`, `EMBED_BATCH_SIZE`, `TAG_PROVIDER`, `TAG_MODEL`, `GEN_PROVIDER`, `GEN_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TAG_SCORING_DEDUP`, `DEDUP_UMBRAL`, `CHROMA_DIR`, `COLLECTION_NAME`, `COSINE_SPACE`, `recreate_index`, `EXPORT_EMBEDDINGS`.
* **Condicionales** (solo si al menos uno de `EMBED/TAG/GEN_PROVIDER` activa el proveedor):
  `OLLAMA_BASE_URL` si hay `ollama`; `HF_DEVICE` si hay `huggingface`. Con `EMBED_PROVIDER=ollama` típica no aparece `HF_DEVICE` ni aporta información.
  - `GOOGLE_API_KEY` y `HF_TOKEN` **nunca** se vuelcan: las credenciales quedan en `.env` por diseño.
* `EMBED_DIM_MAX_*` **no** va en esta tabla: solo es informativa cuando el preflight cayó en caché (red cortada / sin API key), y ese dato vive en la sección 2.
* **Nombres exactos**: la variable que se escribe es la de `config.py` (p. ej. `COSINE_SPACE`, no un alias inventado).
* Los valores que el pipeline ajustó en memoria (p. ej. `EMBED_DIM` corregida al dim del modelo en un AVISO) aparecen ya ajustados.

### 3.3 Sección 2 — Preflight

| Campo | Valores | Interpretación |
|---|---|---|
| `Verificado por` | `online` | Se comprobó en vivo (Ollama `/api/tags` · HF `repo_exists` · Google listado v1beta) |
| `Verificado por` | `cache_env` | No se pudo comprobar online (red cortada / sin API key): se asumió disponible vía `.env` |
| `Dim máxima del modelo` | solo si está declarada | Muestra `EMBED_DIM_MAX_` + el proveedor activo. Si no hay, la línea no se imprime |
| `Aviso` | `sin avisos` o texto | Con `cache_env`, el texto indica qué variable se usó y si está vacía |

### 3.4 Secciones 3–5 — Métricas por fase, chunking e índice

* **Fase a fase:** `FIN` y `INDEX` siempre tienen tiempo; las demás fases con tiempo redondeado a 0 (ejecuciones muy rápidas) muestran `—`. El resumen textual se imprime siempre. Los tiempos se miden con `time.perf_counter()` (precisión real) y se formatean de forma legible: **milisegundos para < 1 s** (p. ej. `245.7 ms`), segundos con 2 decimales para el resto. Evita el "falso 0.0 s".
* **Chunking:** la p25–p75 estrecha indica corpus homogéneo; una cola larga (max mucho mayor que p75) suele ser CSV con filas desbalanceadas. `chunks cortos (< 50) > 0` es ruido probable.
* **Índice final:**
  - `vectores totales` > `vectores insertados` → la colección ya tenía vectores de ejecuciones previas sin `--recreate-index` (acumulación).
  - `Recreado: True/False` = si `--recreate-index` se usó.

### 3.5 Sección 6 — TSD (scoring + dedup)

Si `TAG_SCORING_DEDUP=false`, la sección indica explícitamente *Bloque TSD desactivado* y la interpretación no aplica. Con el bloque activo:

* **Scoring:**
  - `mín / media / máx` de `semantic_score` en [0, 1] (fórmula en *Criterios de scoring* del `README.md`).
  - `chunks con score ≥ 0.6`: aparece como **recuento absoluto `N de total (pct %)`** (p. ej. `1 de 16214 (0.0062 %)`) para que el recuento absoluto no se diluya en un porcentaje redondeado a 0.0. Es la métrica comparable entre ejecuciones.
  - `centralidad media`: cohesión del corpus (coseno vs. centroide); baja si el corpus es muy heterogéneo.
  - `redundancia media`: similitud media con el vecino más cercano (FAISS `k=2`); alta si hay mucho contenido repetido.
  - `tiempo`: fase SCORING completa (array + centroide + redundancia FAISS + bucle), medida con `time.perf_counter()`.
* **Deduplicación:**
  - `descartados (política exacta)`: eliminados por clave de entidad/clave de grupo (CSVs con ID), no por similitud. Con corpus sin CSVs, es 0.
  - `descartados (semánticos)`: eliminados por superar `DEDUP_UMBRAL` de coseno.
  - `total descartados (pct)`: el efecto conjunto; es el nº a vigilar entre ejecuciones.
  - `tiempo`: fase DEDUP completa (mismo formato ms/s).
* **Cobertura por categoría (6.3):** solo aparece con el bloque TSD activo (si no hay etiquetado no existe `doc_category`). Tabla fija de las 6 `doc_category` de la taxonomía cerrada (las que no aparecen en el corpus se muestran como 0, no se ocultan) con:
  - `pre`: nº de chunks de esa categoría antes del dedup.
  - `post (dedup)`: nº de chunks de esa categoría conservados; es la métrica de decisión — si una categoría queda a 0 post-dedup, las preguntas reales de esa intención no podrán responderse (añade corpus o revisa `DEDUP_UMBRAL`).
  - `top 3 tags (post)`: fila extra compacta (máximo 3 tags por nº de chunks conservados, empates alfabéticos) que resume la multi-facialidad *sin inflar* la tabla; el detalle por tag ya se imprime en la consola `[TAG]` por fuente.

### 3.6 Sección 7 — Señales (condición → acción recomendada)

| Señal | Condición en el código | Acción |
|---|---|---|
| **Chunking:** N chunks < 50 | `chunk_stats.cortos > 0` | Revisar los loaders o subir `CHUNK_SIZE` |
| **Dedup:** > 50 % descartado | `dedup.descartados_pct > 50` | Bajar `DEDUP_UMBRAL`; si es lo deseado, no hay que hacer nada |
| **Dedup:** 0 descartados | `dedup.descartados_total == 0` | Subir `DEDUP_UMBRAL` (o aceptar el corpus como es) |
| **Dimensiones:** AVISO/INFO | `dim_msg` no vacío | AVISO: `EMBED_DIM` > dim del modelo → el índice quedó ajustado a la dim del modelo; revisar `.env` e índice. INFO: `EMBED_DIM` < dim máxima del modelo → indexas por debajo del máx.; normal, pero revisa si quieres subir `EMBED_DIM` |
| **Preflight:** caché | `verificado_por == "cache_env"` | La dim mostrada solo es fiable si `EMBED_DIM_MAX_*` está declarada; arranca el servicio o declara la dim para blindar |
| **Scoring:** distribución bajo 0.6 | `score_buenos_pct < 50` | Señal interna de priorización (fórmula propia), **no** precisión de retrieval: validar con consultas reales. Si el corpus se nota pobre, revisa el modelo de `TAG` (solo ve los primeros 6000 caracteres) |
| **Cobertura:** categoría vacía post-dedup | `chunks_por_categoria[cat].post == 0` (por categoría de las 6 cerradas) | Si `pre == 0`: el corpus no cubre esa intención → añade corpus. Si `pre > 0`: el umbral la descartó por completo → baja `DEDUP_UMBRAL` (o acepta que esa intención no se responde) |

Si ninguna se dispara: *Sin señales de alerta: las métricas están dentro de rangos habituales.*

### 3.7 Ejemplo (extracto real)

Ejecución de prueba sobre un corpus de 2 TXT, proveedor `ollama`, bloque TSD activo:

```text
# Informe de indexación

- **Fecha:** 2026-09-14 12:32:46
- **Corpus (rutas):** <tmp>/corpus
- **Tiempo total:** 240.2 ms

## 1. Parámetros aplicados

| Variable | Valor | Nota |
| EMBED_PROVIDER | ollama | proveedor de embeddings |
| EMBED_MODEL | qwen3-embedding:4b | modelo de embeddings |
| EMBED_DIM | 4 | Tope máximo de dimensión (min(dim modelo, EMBED_DIM)) |
| ...
| COSINE_SPACE | cosine | métrica de similitud |
| recreate_index | True | la colección se borró antes de indexar |
| ...(17 filas en total)

## 2. Preflight

- **Verificado por:** `online`
- **Dim máxima del modelo (`EMBED_DIM_MAX_*`):** 4
- **Aviso:** sin avisos
```

En la 6, el `≥ 0.6` lleva recuento absoluto (el `0.01 %` real aparece como `N de total (pct %)`) y los tiempos en ms/s:

```text
### 6.1 Scoring (semantic_score)

| Métrica | Valor | Interpretación |
| chunks con score ≥ 0.6 | 2 de 2 (100.0 %) | cuota de chunks 'útiles' según el modelo |
| tiempo | 2.1 ms | fase SCORING |
```

Y, como el dim resultante (4) supera el `EMBED_DIM` declarado en `.env` (2560):

```text
## 5. Índice final (ChromaDB)

- **Dim de embedding:** 4
- **Vectores insertados:** 1
- **Recreado (`--recreate-index`):** True

## 7. Señales y criterios de decisión

- **Dimensiones:** EMBED_DIM (2560) supera la dim máxima del modelo (4):
  EMBED_DIM ajustado a 4 — las dimensiones con las que se crea el índice
  son el máximo que maneja el modelo.
```

## 4. `output/embeddings.json` (opcional)

Controlado por `EXPORT_EMBEDDINGS` (`.env`, default `true`). Si está activo, al terminar `INDEX` se escribe `output/embeddings.json` con `text + metadata + embedding` por chunk (para inspección y depuración). Es un dump crudo: no sustituye al informe.

## 5. Validación y pruebas

* `tests/test_informe.py` — renderización unitaria: tabla de parámetros, métricas de fase/chunk, señales con dedup activo, salida en `output/` por defecto, métricas de runtime de scoring/dedup.
* `tests/test_informe_e2e.py` — humo extremo a extremo con proveedor fake: verifica que el dict devuelto por `ejecutar_pipeline` contiene todas las claves que consume el informe y que el markdown por defecto sale en `output/informe_index_aaaaMMdd_hhmm.md`.

Ejecutar: `python -m pytest tests/test_informe.py tests/test_informe_e2e.py -v`
