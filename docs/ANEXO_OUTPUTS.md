# Guía técnica: salidas del pipeline (informe de indexación)

## 1. Objeto

Este documento describe las salidas que genera la función `ejecutar_pipeline` del directorio de salidas, y, sobre todo, la **estructura y la interpretación del informe de indexación** (el archivo `output/informe_index_aaaaMMdd_hhmm.md`): qué cabe esperar en cada sección, qué reglas gobiernan la tabla de parámetros y cómo leer las señales automáticas para decidir qué conviene ajustar.

## 2. Alcance y referencias

- **Aplicabilidad**: el módulo `src/pipeline.py` (el que ensambla el diccionario de métricas), el módulo `src/informe.py` (el que renderiza el documento de markdown) y el módulo `src/index.py` (el que aporta los datos de la quinta sección).
- **Generación**: cada ejecución del pipeline escribe por defecto el archivo `output/informe_index_aaaaMMdd_hhmm.md` (con la fecha y la hora locales de la ejecución), salvo cuando se desactiva con `informe=False` en las pruebas. No existe un flag de consola al efecto: el informe se genera siempre al terminar.
- **Referencias**: la sección *Uso* del archivo `README.md` y la relación de variables del archivo `.env` (las variables que aparecen en el informe).
- **Validación**: la prueba `tests/test_informe.py` (renderizado unitario, sin red) y la prueba `tests/test_informe_e2e.py` (prueba de humo de extremo a extremo con proveedor falso).

## 3. El informe de indexación (archivo `output/informe_index_aaaaMMdd_hhmm.md`)

El objetivo del informe es responder, en una sola lectura, **qué parámetros determinaron esta ejecución** y **de qué calidad quedó la indexación**, para tomar decisiones (bajar o subir el umbral, cambiar de modelo, regenerar el índice).

### 3.1 Estructura (siete secciones, siempre en el mismo orden)

- **La primera sección (Parámetros aplicados)**: una tabla de variable, valor y nota, con los parámetros que influyeron en esta ejecución. Sirve para verificar la configuración efectiva; incluye las variaciones aplicadas por flag (por ejemplo, un `--chunk-size 500` aparece como el valor cien cincuenta aunque la constante `CHUNK_SIZE` del archivo `.env` sea otro).
- **La segunda sección (Preverificación)**: el indicador de por quién se verificó el modelo (el valor `online` o el valor de memoria caché del entorno), la dimensión máxima declarada (si la hay) y el aviso. Sirve para detectar ejecuciones degradadas (red cortada o ausencia de clave API) antes de confiar la dimensión.
- **La tercera sección (Métricas por fase)**: el tiempo en segundos y el resumen de cada fase, desde la preverificación hasta la terminación. Sirve para localizar la fase lenta o la etapa en la que se descartaron chunks.
- **La cuarta sección (Corpus y chunking, longitudes en caracteres)**: las definiciones de fuente/documento/chunk, la tabla de fuentes (archivos) con sus documentos y chunks pre/post dedup (los tres niveles del corpus), la longitud mínima, el percentil veinticinco, la media, el percentil cincuenta, el percentil setenta y cinco y el máximo (en **caracteres**), el recuento de chunks cortos (menores de cincuenta) y el recuento de íntegras (entidades `no_chunk`, sin trocear). Sirve para evaluar si el troceado es razonable y de dónde viene cada chunk.
- **La quinta sección (Índice final, en ChromaDB)**: la colección, el espacio de similitud, la dimensión, los vectores insertados frente a los totales de la colección, si se recreó y la **integridad** de los vectores (dimensión homogénea, NaN/Inf, ceros, norma L2 mínima/media/máxima). Sirve para confirmar que la inserción quedó verificada y que el tamaño y los vectores son los esperados.
- **La sexta sección (El bloque TSD, de puntuación y deduplicación)**: 6.1 la distribución de la puntuación semántica (con glosario de métricas y tabla de redundancia por política de dedup), 6.2 el desglose de los descartados (los exactos frente a los semánticos) y 6.3 la **cobertura por categoría** (antes y después de la deduplicación) así como los tres tags principales. Sirve para medir la densidad de señal del índice, el efecto del umbral de deduplicación y qué intenciones quedan cubiertas.
- **La séptima sección (Señales y criterios de decisión)**: las heurísticas automáticas interpretadas (la relación figura en el apartado 3.6). Sirve para saber qué revisar o ajustar a partir de esta ejecución.

También existe un encabezado común a todas las secciones: la fecha de ejecución (UTC local), las rutas del corpus y el tiempo total.

### 3.2 La primera sección: parámetros aplicados

Las reglas de la tabla (implementadas en `src/pipeline.py`) son las siguientes:

- **Siempre se muestran**, por ser irrelevantes al proveedor y afectar a esta ejecución: las constantes `EMBED_PROVIDER`, `EMBED_MODEL`, `EMBED_DIM_DECLARADO`/`EMBED_DIM_MODELO`/`EMBED_DIM` (las tres filas de dimensión), `EMBED_BATCH_SIZE`, `TAG_PROVIDER`, `TAG_MODEL`, `GEN_PROVIDER`, `GEN_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TAG_SCORING_DEDUP`, `DEDUP_UMBRAL`, `CHROMA_DIR`, `COLLECTION_NAME`, `COSINE_SPACE`, el indicador `recreate_index` y la constante `EXPORT_EMBEDDINGS`.
- **Se muestran de forma condicional**, únicamente cuando al menos uno de los tres proveedores (de embeddings, etiquetado o generación) activa el proveedor en cuestión: la constante `OLLAMA_BASE_URL` si hay Ollama, y la constante `HF_DEVICE` si hay HuggingFace. Con un proveedor típico de embeddings por Ollama, la constante `HF_DEVICE` no aparece ni aporta información.
- **Nunca se vuelcan** las credenciales: la constante `GOOGLE_API_KEY` y el token `HF_TOKEN` quedan siempre en el archivo `.env`, por diseño.
- **Las tres filas de dimensión** se muestran por separado: `EMBED_DIM_DECLARADO` (el tope del archivo `.env`), `EMBED_DIM_MODELO` (la dimensión máxima declarada en la preverificación, `EMBED_DIM_MAX_*`: con el valor declarado o, cuando consta vacía en el archivo `.env`, el texto `no declarada`, porque la comprobación en línea solo verifica la disponibilidad del modelo y nunca su dimensión nativa) y `EMBED_DIM` (la dimensión efectiva del índice, la de los vectores generados). No hay ningún ajuste en memoria: la tabla refleja los tres valores reales de la ejecución.
- **La constante `EMBED_DIM_MAX`** no va en esta tabla: únicamente es informativa cuando la preverificación recurre a la memoria caché (red cortada o ausencia de clave API), y ese dato vive en la segunda sección.
- **Nombres exactos**: la variable que se escribe es la de `config.py` (por ejemplo, la constante `COSINE_SPACE`, no un alias inventado).

### 3.3 La segunda sección: preverificación

- **El campo «Verificado por»**, con el valor `online`: significa que se comprobó en vivo (el punto `/api/tags` de Ollama, la comprobación `repo_exists` de HuggingFace, o el listado de Google de la versión beta uno).
- **El campo «Verificado por»**, con el valor de memoria caché: significa que no fue posible comprobar en línea (red cortada o ausencia de clave API); se asumió disponible mediante el archivo `.env`.
- **El campo «Dimensión máxima del modelo»**: cuando está declarada aparece el valor de la constante `EMBED_DIM_MAX_<proveedor>`; cuando no lo está, la línea se imprime con el texto `no declarada`, para dejar constancia de que la comprobación en línea no expone la dimensión nativa del modelo y de que esa dimensión únicamente procede de la declaración manual.
- **El campo «Aviso»**: su valor es `sin avisos` o un texto. Con el valor de memoria caché, el texto indica qué variable se usó y si está vacía.

### 3.4 Las secciones tres, cuatro y cinco: métricas por fase, troceado e índice

- **Fase a fase**: las fases de terminación y de índice siempre tienen tiempo; las demás fases, cuando su tiempo arredondeado es cero (ejecuciones muy rápidas), muestran un guion largo. El resumen textual se imprime siempre. Los tiempos se miden con la función del temporizador de alta resolución del runtime de Python y se formatean de manera legible: se muestran en **milisegundos cuando son inferiores a un segundo** (por ejemplo, `245.7 ms`) y, en el resto de los casos, en segundos con dos decimales. Con esto se evita un «falso cero de segundos».
- **Corpus y chunking**: los chunkings se miden en **caracteres** (la función de longitud de la partición es `len`). La tabla de fuentes muestra los tres niveles del corpus (fuentes → documentos → chunks pre/post). Una distancia estrecha entre el percentil veinticinco y el setenta y cinco indica un corpus homogéneo; una cola larga (un máximo muy superior al percentil setenta y cinco) suele ser un archivo CSV con filas desbalanceadas. Un recuento de chunks cortos (menores de cincuenta) mayor de cero constituye ruido probable. El máximo puede superar `CHUNK_SIZE` sin ser un error: son las filas de entidad indexadas íntegras (una fila = un chunk, sin trocear).
- **Índice final**: si los `vectores totales` son mayores que los `vectores insertados`, la colección ya contenía vectores de ejecuciones previas que no se recrearon (acumulación). El campo «Recreado», que toma el valor verdadero o falso, indica si se usó el flag de recrear índice. La línea **Integridad** (solo con vectores) resume la comprobación de los vectores finales: dimensión homogénea, sin NaN/Inf ni ceros, y la norma L2 mínima/media/máxima (los vectores salen normalizados: la norma debe estar en 1).

### 3.5 La sexta sección: bloque TSD (puntuación y deduplicación)

Cuando la constante `TAG_SCORING_DEDUP` está desactivada, la sección indica explícitamente *bloque TSD desactivado* y la interpretación que sigue no aplica. Con el bloque activo:

- **En cuanto a la puntuación**:
  - El mínimo, la media y el máximo de la puntuación semántica, entre cero y uno (la fórmula figura en la sección *Criterios de puntuación semántica* del `README.md`).
  - La métrica de **chunks con puntuación mayor o igual que 0.6**: aparece como **recuento absoluto** (un número `N` sobre el total, con su porcentaje; por ejemplo, `1 de 16214 (0.0062 %)`), de modo que el recuento absoluto no se diluya en un porcentaje redondeado a cero. Es la métrica comparable entre ejecuciones.
  - **La centralidad media**: mide la cohesión del corpus (la similitud coseno frente al centroide); resultará baja si el corpus es muy heterogéneo.
  - **La redundancia media**: es la similitud media con el vecino más cercano (una búsqueda con FAISS de dos vecinos); resultará alta si hay mucho contenido repetido.
  - **El tiempo**: el de la fase de puntuación completa (la construcción de la matriz, el centroide, la redundancia con FAISS y el bucle), medido con la función del temporizador de alta resolución del runtime de Python.
- **En cuanto a la deduplicación**:
  - **Los descartados por política exacta**: los eliminados por clave de entidad o de grupo (los archivos CSV con identificador), no por similitud. Con un corpus sin archivos CSV, su recuento es cero.
  - **Los descartados semánticos**: los eliminados por superar el umbral de deduplicación de similitud coseno.
  - **El total de descartados (con su porcentaje)**: el efecto conjunto; es el recuento que conviene vigilar entre ejecuciones.
  - **El tiempo**: el de la fase de deduplicación completa, con el mismo formato de milisegundos o segundos.
- **La cobertura por categoría (la sexta sección tercera)**: únicamente aparece con el bloque TSD activo (si no hay etiquetado no existe la categoría del documento). Es una tabla fija de las seis categorías de la taxonomía cerrada (las que no aparecen en el corpus se muestran como cero, sin ocultarse), con:
  - **El recuento previo**: el número de chunks de esa categoría antes de la deduplicación.
  - **El recuento posterior (tras la deduplicación)**: el número de chunks de esa categoría conservados; es la métrica de decisión, pues si una categoría queda a cero, las preguntas reales de esa intención no podrán responderse (conviene añadir corpus o revisar el umbral de deduplicación).
  - **Los tres tags principales (post)**: una fila adicional compacta (los máximo tres tags por número de chunks conservados, con los empates en orden alfabético) que resume la multi-facialidad *sin inflar* la tabla; el detalle por tag ya se imprime en la consola del registro de etiquetado por fuente.

### 3.6 La séptima sección: señales (condición y acción recomendada)

- **Señal de troceado** (N chunks menores de cincuenta): se dispara cuando el recuento de chunks cortos en los estadísticos es mayor de cero. Acción: revisar los cargadores o subir la constante de tamaño de chunk.
- **Señal de deduplicación** (descartada más del cincuenta por ciento): se dispara cuando el porcentaje de descartados en la métrica de deduplicación es mayor de cincuenta. Acción: bajar el umbral de deduplicación; si así se desea, no hace falta nada.
- **Señal de deduplicación** (cero descartados): se dispara cuando el total de descartados en la métrica de deduplicación es cero. Acción: subir el umbral de deduplicación (o aceptar el corpus tal cual es).
- **Señal de fuente vacía (crítica)** (un archivo entero a cero chunks tras la deduplicación): se dispara cuando una fuente de la tabla del apartado cuatro tiene chunks previos y cero posteriores. Es la señal más grave del informe: el contenido de todo ese archivo deja de ser recuperable (por ejemplo, un PDF de una sola página cuyo único chunk se descartó). Acción: revisar el umbral de deduplicación o regenerar el índice; con una fuente de un solo chunk el descarte suele ser azar de la deduplicación semántica, ya que la puntuación semántica varía entre ejecuciones (el etiquetado con el modelo de lenguaje no es determinista).
- **Señal de dimensiones** (aviso o informativo): se dispara cuando el mensaje de dimensión no está vacío. Con un aviso (`EMBED_DIM` declarado mayor que la dim generada), el índice se crea con la dim generada (las dimensiones del modelo); conviene revisar el valor de `EMBED_DIM` en el archivo `.env`. Con un mensaje informativo (la dimensión declarada está por debajo de la dimensión máxima del modelo), se indexa por debajo del máximo; es un estado normal, pero conviene revisar si se desea subir la constante de dimensión.
- **Señal de preverificación** (memoria caché): se dispara cuando el verificado por es el valor de memoria caché. La dimensión mostrada únicamente resulta fiable si la constante `EMBED_DIM_MAX` lleva declarada; conviene arrancar el servicio o declarar la dimensión para blindar la comprobación.
- **Señal de puntuación** (distribución por debajo de 0.6): se dispara cuando el porcentaje de chunks con buena puntuación es inferior a cincuenta. Es una señal interna de priorización (una fórmula propia), y **no** mide la precisión de la recuperación: conviene validar con consultas reales. Si el corpus se nota pobre, conviene revisar el modelo del etiquetado (que únicamente ve los primeros 6000 caracteres).
- **Señal de cobertura** (categoría vacía tras la deduplicación): se dispara cuando el total posterior de alguna de las seis categorías cerradas es cero. Si el previo es cero, el corpus no cubre esa intención y conviene añadir corpus; si el previo es mayor de cero, fue el umbral el que la descartó por completo y conviene bajar el umbral (o aceptar que esa intención no se responde).

Cuando ninguna señal se dispara, el informe indica: *Sin señales de alerta: las métricas están dentro de rangos habituales*.

### 3.7 Ejemplo (extracto real)

Ejecución de prueba sobre un corpus de dos archivos TXT, con proveedor Ollama y con el bloque TSD activo:

```text
# Informe de indexación

- **Fecha:** 2026-09-14 12:32:46
- **Corpus (rutas):** <tmp>/corpus
- **Tiempo total:** 240.2 ms

## 1. Parámetros aplicados

| Variable | Valor | Nota |
| `EMBED_PROVIDER` | ollama | proveedor de embeddings |
| `EMBED_MODEL` | qwen3-embedding:4b | modelo de embeddings |
| `EMBED_DIM_DECLARADO` | 3072 | tope declarado en .env |
| `EMBED_DIM_MODELO` | 2560 | dimensión máxima del modelo (EMBED_DIM_MAX_*) |
| `EMBED_DIM` | 4 | dimensión efectiva del índice (min(dim modelo, declarado)) |
| ...
| `COSINE_SPACE` | cosine | métrica de similitud |
| `recreate_index` | True | la colección se borró antes de indexar |
| ...(18 filas en total)

## 2. Preflight

- **Verificado por:** `online`
- **Dim máxima del modelo (`EMBED_DIM_MAX_*`):** 1024
- **Aviso:** sin avisos
```

En la sección cuarta, los largos se miden en caracteres y la tabla de fuentes muestra los tres niveles (fuentes → documentos → chunks):

```text
## 4. Corpus y chunking (longitudes en caracteres)

| fuente | docs | chunks pre | chunks post (dedup) |
|---|---|---|---|
| a.txt | 1 | 1 | 1 |
| **total** | 2 | 3 | 2 |

| Métrica (caracteres) | Valor |
|---|---|
| min | 10 |
| ...
| íntegras (`no_chunk`) | 0 |

> Un `max` mayor que `CHUNK_SIZE` no es un error: son las entidades indexadas íntegras (una fila = un chunk).
```

En la sección sexta, la métrica de puntuación mayor o igual que 0.6 lleva recuento absoluto (el valor real del 0.01 por ciento aparece en el formato de «un número, sobre el total, con su porcentaje») y se documenta como umbral interno, y los tiempos se muestran en milisegundos o segundos:

```text
### 6.1 Scoring (`semantic_score`)

| Métrica | Valor | Interpretación |
|---|---|---|
| chunks con score ≥ 0.6 | 1 de 2 (50.0 %) | umbral interno de esta fase: NO es precisión ni relevancia frente a consultas |
| tiempo | 2.1 ms | fase SCORING |
```

Y, como `EMBED_DIM` declarado (3072) es mayor que la dim generada (4), la señal de dimensiones avisa de que el índice se creó con la dim generada:

```text
## 5. Índice final (ChromaDB)

- **Dim de embedding:** 4
- **Integridad:** 1 vectores · dim 4 · NaN/Inf 0 · ceros 0 · norma L2 mín 1.0000 / media 1.0000 / máx 1.0000

## 7. Señales y criterios de decisión

- **Dimensiones:** EMBED_DIM (3072) es mayor que la dim generada (4): el índice se crea con 4 dims — la dim del índice son las 4 dimensiones que genera el modelo.
```

## 4. El archivo `output/embeddings.json` (opcional)

Está controlado por la constante `EXPORT_EMBEDDINGS` (en el archivo `.env`, con el valor por defecto activo). Cuando está activo, al terminar la fase de indexación se escribe el archivo `output/embeddings.json` con el texto, los metadatos y el vector de cada chunk (para inspección y depuración). Se trata de un volcado crudo: no sustituye al informe.

## 5. Validación y pruebas

- La prueba `tests/test_informe.py`: el renderizado unitario, la tabla de parámetros, las métricas de fase y de troceado, las señales con la deduplicación activa, la salida en el directorio de salidas por defecto, y las métricas de tiempo de ejecución de la puntuación y la deduplicación.
- La prueba `tests/test_informe_e2e.py`: la prueba de humo de extremo a extremo con proveedor falso; verifica que el diccionario devuelto por la función del pipeline contiene todas las claves que consume el informe y que el documento de markdown por defecto sale en `output/informe_index_aaaaMMdd_hhmm.md`.

Ejecutar ambas pruebas: `python -m pytest tests/test_informe.py tests/test_informe_e2e.py -v`
