# FT_CORPUS_DATA — Estado completo del pipeline RAG de deporte municipal

Referencia del estado actual del pipeline offline de corpus e indexación: qué hace el programa, el detalle de cada módulo, los proveedores y su configuración, el flujo de invocación, las métricas por consola y la escala de niveles de troceado, junto con la fase online pendiente. Para las firmas y el comportamiento de cada función, consultar [`ANEXO_FUNC.md`](./ANEXO_FUNC.md); para las salidas generadas (el informe de indexación y el volcado de embeddings), [`ANEXO_OUTPUTS.md`](./ANEXO_OUTPUTS.md); y para el procedimiento de cambio del nivel de troceado, [`ANEXO_CHUNKING.md`](./ANEXO_CHUNKING.md).

## 1. Qué hace el programa (estado actual)

Se trata de un pipeline offline de Recuperación Aumentada por Generación (RAG) que prepara un índice vectorial listo para una recuperación (retrieval) efectiva. El orquestador (`src/pipeline.py`) encadena las fases en este orden: preverificación (PREFLIGHT), carga (LOAD), limpieza (CLEAN), etiquetado con modelo de lenguaje (TAG), troceado (CHUNK), vectorización (EMBED), puntuación semántica (SCORING), deduplicación (DEDUP), indexación (INDEX) y generación del informe (INFORME).

- **La preverificación de disponibilidad** (la función `verificar_modelo_disponible` en `src/embed.py`, fase PREFLIGHT, situada antes de la carga): comprueba que el modelo indicado por la constante `EMBED_MODEL` esté disponible en el proveedor indicado por la constante `EMBED_PROVIDER` (mediante la ruta `/api/tags` en Ollama, la comprobación `repo_exists` en HuggingFace o el listado de modelos en Google) y **se interrumpe** (lanza un error de tiempo de ejecución) cuando no lo está. Si la comprobación en línea no es posible (red cortada o ausencia de clave API), recurre a la memoria caché offline declarada en las constantes `EMBED_DIM_MAX_*`, con un aviso por consola.
- **El núcleo clásico** (los módulos `load.py`, `clean.py`, `chunk.py`, `embed.py` e `index.py`, junto con la capa CSV): carga archivos PDF, TXT, MD y CSV (archivo único o carpeta recursiva), normaliza el texto, trocea mediante el divisor recursivo por caracteres (el nivel dos, que es el actual), genera los embeddings de forma multi-proveedor y los persiste en ChromaDB (métrica coseno).
- **La capa CSV** (los módulos `csv_advisor.py` y `csv_transform.py`): aplica una clasificación diferencial a cada archivo CSV (entidad con identificador estable, tabla de hechos, contenido textual o archivo desconocido) y decide el tratamiento y la política de deduplicación de cada fuente; la deduplicación de los archivos CSV **nunca es semántica**, sino por clave exacta. El detalle de las funciones, de la decisión aplicada al corpus actual y del glosario de los registros está en la tercera sección (la capa CSV) de [`ANEXO_FUNC.md`](./ANEXO_FUNC.md).
- **El bloque TSD** (etiquetado, puntuación y deduplicación; los módulos `tsd/tag.py`, `tsd/scoring.py` y `tsd/dedup.py`): etiquetado semántico con un modelo de lenguaje bajo una taxonomía cerrada, puntuación de cada chunk (relevancia, centralidad, redundancia y autoridad) y deduplicación recorte-greedy por similitud coseno. Su objetivo es que al índice entren únicamente chunks bien puntuados y no redundantes, de manera que la recuperación sea más precisa y la respuesta final más fiable. El bloque es opcional: se activa y desactiva con la constante `TAG_SCORING_DEDUP`.
- **La abstracción multi-proveedor**: los embeddings y el modelo de lenguaje pueden salir de **Ollama** (offline, sin coste), de **HuggingFace** (sentence-transformers sobre la biblioteca transformers) o de **Google** (la API de servicio web de Gemini). Se cambia mediante variables de entorno (`EMBED_PROVIDER`, `TAG_PROVIDER` y `GEN_PROVIDER`), sin tocar el código.
- **El registro por consola**: cada fase registra su avance y sus métricas (números de documentos, chunks y vectores; la distribución de longitudes mínima, percentil veinticinco, media, percentil setenta y cinco y máxima; la dimensión de los embeddings; los scores mínimo, medio y máximo; los chunks descartados por la deduplicación; el tiempo total).

### 1.1 Cómo lanzar todo el pipeline

Para lanzar el pipeline completo (desde la preverificación hasta la indexación y el informe), desde el directorio raíz del proyecto:

```powershell
python -m src.pipeline data --recreate-index
```

Antes, es preciso crear el entorno virtual de Python e instalar las dependencias:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

La línea de comandos admite, además de los valores definidos en el archivo `.env`, los siguientes parámetros:

```text
python -m src.pipeline [-h] [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP]
    [--persist-dir PERSIST_DIR] [--collection COLLECTION] [--recreate-index] rutas [rutas ...]
```

- `rutas`: los archivos o las carpetas a indexar (por ejemplo, `data`); es obligatorio indicarlos.
- `--chunk-size`: la longitud del chunk; por omisión, la constante `CHUNK_SIZE` del archivo `.env`.
- `--chunk-overlap`: el solapamiento entre chunks; por omisión, la constante `CHUNK_OVERLAP` del archivo `.env`.
- `--persist-dir`: el directorio donde reside ChromaDB; por omisión, la constante `CHROMA_DIR` del archivo `.env`.
- `--collection`: el nombre de la colección; por omisión, la constante `COLLECTION_NAME` del archivo `.env`.
- `--recreate-index`: borra la colección antes de indexar; por omisión, no se borra.

## 2. Los módulos

### `src/load.py` — la carga

- **Los cargadores por extensión**: los archivos `.pdf` se cargan con el cargador de PDF (la biblioteca PyPDF); los archivos `.txt` y `.md` con el cargador de texto plano; y los archivos `.csv` se convierten en documentos (una fila por documento, o varios según la clasificación del asesor) mediante el módulo `csv_transform`.
- La función pública `cargar_archivos` (parámetro: una ruta única o una lista de rutas) admite un único archivo o una o varias carpetas (recursivas) y devuelve una lista de documentos con los metadatos `source` (el nombre base del archivo) y `file_hash` (el resumen SHA-256 del archivo crudo) **siempre poblados**, aunque el cargador no los haya fijado. Los archivos cuya extensión no está soportada se omiten dejando constancia por consola.

### `src/clean.py` — la limpieza

La función `limpiar` (parámetro: una lista de documentos o una cadena) normaliza el texto antes de trocearlo: colapsa los saltos de línea múltiples, colapsa los espacios y tabulaciones sobrantes, elimina las líneas vacías repetidas y suprime los caracteres de control y el carácter de orden al inicio (BOM). No toca los metadatos: devuelve una copia superficial del diccionario de metadatos de cada documento.

### `src/chunk.py` — el troceado (nivel dos)

Trocea el texto de cada documento en fragmentos de tamaño acotado, rompiendo siempre **primero en los límites más naturales** y solo descendiendo cuando hace falta:

1. Emplea el **divisor recursivo de texto por caracteres** con los separadores de párrafo, salto de línea, frase, espacio y letra: corta en párrafos antes que en frases, en frases antes que en palabras y en palabras antes que en letras. De este modo, un chunk típico empieza y termina en un límite semántico, no a mitad de palabra.
2. La función pública `trocear` (parámetros: una lista de documentos o una cadena, y opcionalmente el tamaño y el solapamiento de chunk) es la interfaz: acepta una lista de `Document` o una cadena de texto aislada (para pruebas y scripts). A cada chunk añade la clave `chunk_index` (secuencial **por documento**, desde cero) y la clave `chunk_size`, y copia la metadata del documento padre.
3. Por eso constituye el **puente entre las fases de etiquetado y de troceado**: todo lo que la fase de etiquetado escribió en la metadata de la fuente (la categoría del documento, los tags, la relevancia asignada por el modelo y los booleanos de cada tag) viaja con cada chunk hasta el índice, sin que nadie tenga que copiarlo a mano.
4. El módulo, además, **respeta la pista de troceado** emitida por el asesor (la clave `chunking_hint`): con la pista de no trocear, el documento se conserva íntegro como un único chunk (entidad atómica); con la pista de troceado ligero, si su longitud cabe dentro del tamaño de chunk, igualmente no se trocea.

**El alcance de la clave `chunk_index`**: es secuencial de cero a n **dentro de cada documento**, no global (un archivo CSV genera muchos documentos con la misma fuente). El identificador global del índice lo asigna `pipeline.py` y ese sí es único en Chroma (en la sección cuarta se explica el flujo de invocación).

### `src/embed.py` — los vectores (multi-proveedor)

Este módulo es la única puerta de conversión de texto a vectores: tanto la indexación como la consulta futura transitan por ella, de modo que la dimensión se garantiza una única vez.

- La función `embeddear` (parámetros: los textos, y opcionalmente el modelo y la dimensión), según la constante `EMBED_PROVIDER`:
  - Para el proveedor **Ollama**, la ruta por lotes `/api/embed`, en lotes de tamaño `EMBED_BATCH_SIZE`, con el tiempo máximo de espera por petición fijado en la constante `EMBED_TIMEOUT` (el tiempo por lote es el de carga fría del modelo).
  - Para el proveedor **HuggingFace**, el cargador de sentence-transformers (carga perezosa con memoria caché) con la normalización de embeddings activada (coherente con la métrica coseno).
  - Para el proveedor **Google**, el punto de servicio web por lotes (requiere la constante `GOOGLE_API_KEY`).
- La función `embeddear_consulta` (parámetro: la pregunta): la fase online. **La condición RAG ineludible**: el índice y la consulta han de usar el mismo modelo; ofrece idéntica garantía de dimensión que la indexación.
- La función `exportar_json` (parámetros: los chunks y los embeddings, y la ruta, por omisión `output/embeddings.json`): persiste el texto, los metadatos y el vector de cada registro, para inspección y depuración (lo consume el script de coste del bloque TSD).
- La función `verificar_modelo_disponible`: la preverificación de disponibilidad descrita en la sección primera. Si la comprobación en línea no es posible, devuelve el indicador de verificado por con el valor de memoria caché del entorno y recurre a la constante `EMBED_DIM_MAX_` del proveedor activo.

### `src/index.py` — ChromaDB

- Cliente persistente anclado en el directorio `CHROMA_DIR`; la colección se crea con la métrica coseno.
- La función `indexar` (parámetros: los identificadores, los embeddings, los documentos, los metadatos, y opcionalmente el directorio persistente, el nombre de la colección y la opción de recreación): sanea los metadatos (Chroma no admite valores vacíos ni listas: se convierten en la cadena `null` y en cadenas, respectivamente), inserta por lotes y **verifica** lote a lote que los identificadores estén vivos en la colección, devolviendo una tupla con los vivos y el total. Admite la opción de recrear, que borra la colección antes de indexar.
- Las funciones reutilables `obtener_cliente_chroma` (el cliente persistente) y `obtener_guid_chroma` (el identificador único de la base de datos) quedan pensadas para la fase de recovery online; el módulo de informe las consume para nombrar el reporte único por ejecución.

### `src/tsd/tag.py` — el etiquetado (con modelo de lenguaje)

- Etiquetas **una vez por fuente** (una única llamada al modelo de lenguaje por documento fuente, no por página) y propaga el resultado a todos sus documentos, lo que lo hace luego barato y consistente.
- La taxonomía es cerrada:
  - La categoría del documento (`doc_category`) toma sus valores de un conjunto de seis categorías cerradas: tarifas, normativa, reservas, abonos, instalaciones y agenda.
  - Los tags (`tags`) son como máximo ocho de una lista de dieciséis tags válidos; se conservan como una cadena unida por punto y coma, dado que Chroma no admite listas (es de lectura humana).
  - La relevancia asignada por el modelo (`relevancia_llm`) es un valor entre cero y uno.
  - El analizador filtra los valores devueltos contra los conjuntos válidos: una categoría fuera de la taxonomía se sustituye por la de instalaciones (con aviso por consola) y los tags fuera se descartan silenciosamente.
- **Booleanos por tag** (la clave de cada tag con el valor verdadero, escritura escasa: solo se crea la clave del tag presente): constituyen el **contrato de filtrado** de la fase online. El filtro de Chroma (la cláusula `where`) soporta los booleanos y la clave ausente no coincide, de modo que la consulta filtra directamente sobre el índice (por ejemplo, un filtrado compuesto por categoría de reservas y por el booleano de cancelación se escribe como la cláusula con el operador lógico de conjunción y ambas condiciones de igualdad).
- El proveedor es configurable (el interruptor de etiquetado, que hereda por omisión el de generación): Ollama, HuggingFace o Google.
- Robustez: si el modelo devuelve un objeto JSON malformado, se asigna por defecto la categoría de instalaciones en lugar de interrumpir el pipeline.

### `src/tsd/scoring.py` — la puntuación semántica

La puntuación semántica se combina como la suma ponderada de cuatro señales aditivas en el intervalo de cero a uno (recortada y redondeada a cuatro decimales), guardada en el metadato `semantic_score`:

- **La relevancia asignada por el modelo** (factor 0,45): es la relevancia de cero a uno que el modelo asigna a la fuente en la fase de etiquetado (¿responde a preguntas de tarifas, reservas, horarios y normas?). Es el factor dominante porque es la única señal alineada directamente con el objetivo de Recuperación Aumentada por Generación, y no solo con la geometría de los vectores.
- **La centralidad** (factor 0,20): el coseno del chunk contra el centroide de la colección (la media de todos los vectores, normalizada). Recompensa el contenido representativo del corpus y penaliza lo marginal, sin llegar a dominar el score.
- **La no redundancia** (factor 0,20): uno menos el coseno con el vecino más cercano (búsqueda de dos vecinos en FAISS, excluyendo el propio). Penaliza los chunks casi duplicados: si el contenido ya existe en otro chunk, no aporta señal nueva.
- **La autoridad** (factor 0,15): el peso por tipo de fuente (coincidencia por subcadena sobre el nombre de la fuente): reglamento o normativa uno coma cero; precios o tarifas cero coma noventa; agenda cero coma sesenta; el resto cero coma setenta. Ajuste fino: una norma vale más que un archivo CSV genérico, pero no puede compensar una baja relevancia.

- La puntuación alimenta el recorte greedy de la fase DEDUP y, en el mapa de ruta, la reordenación de la recuperación.
- Se usa **FAISS** (el índice plano de producto interno, búsqueda de dos vecinos) para la redundancia en vez de materializar la matriz completa de similitud: el consumo de memoria pasa de ser proporcional al cuadrado del número de filas a ser proporcional a las filas por la dimensión (aproximadamente 1,3 GB frente a los 68 GB en la escala real del proyecto), que era la causa del agotamiento de memoria en la fase del bloque TSD.

### `src/tsd/dedup.py` — la deduplicación

Ejecuta una deduplicación **consciente de política**:

- **Por política exacta** (chunks con una política de deduplicación estricta, es decir los de archivos CSV): los chunks se deduplican **únicamente por su clave de identidad** y **nunca por similitud coseno** (de este modo dos entidades casi idénticas, como dos piscinas del mismo barrio que difieren en pocos datos, no se descartan como duplicados semánticos).
- **Por recorte semántico greedy** (el resto de los chunks): ordenados por la puntuación semántica descendente, se descarta un chunk cuando su coseno con alguno de los chunks ya conservados es **mayor o igual** al umbral de deduplicación (el valor por omisión es 0,93; conviene subir a 0,95 si se deduplica demasiado y bajar a 0,90 si queda todavía redundancia).

La implementación es **incremental** sobre **FAISS** (el índice plano): cada candidato se busca contra los chunks ya conservados (un vecino), en vez de reconstruir la matriz completa de similitud (el mismo agotamiento de memoria que en la fase de puntuación). Su semántica es idéntica a la matriz, pero con un consumo de memoria lineal, acotado por el número de chunks conservados.

### `src/pipeline.py` — el orquestador

La función `ejecutar_pipeline` encadena todas las fases y devuelve un diccionario de métricas con las siguientes claves:

- `num_documentos_cargados`, `num_documentos_normalizados` y `num_documentos`.
- `num_chunks_pre_dedup` (los chunks antes de deduplicar) y `num_chunks_post_dedup` (los chunks después de deduplicar).
- `chunks_descartados` (la diferencia entre ambos).
- `dim_embedding` (la dimensión de los vectores).
- `chunk_stats`: los estadísticos de longitud (el mínimo, el percentil veinticinco, la media, el percentil setenta y cinco y el máximo) y el recuento de chunks cortos (menores de cincuenta caracteres).
- `scoring`: la métrica de puntuación (mínimo, media y máximo), la centralidad media, la redundancia media y el recuento de chunks con score bueno.
- `dedup`: el umbral, los recuentos previos y posteriores, los descartados exactos y semánticos, el porcentaje y la **cobertura por categoría** (las seis categorías cerradas, previa y posterior a la deduplicación) junto con `tags_top3_post` (los tres tags principales del posterior).
- `indice`: los datos de la colección final (nombre, espacio, dimensión, vectores insertados frente a totales y si se recreó).
- `tiempo_total_s`: el tiempo total en segundos.

Devuelve, además, el diccionario que consume el módulo de informe (`parametros`, `fases`, `resumen_fases`, `preflight`), que constituye el contrato entre el pipeline y el informe.

## 3. Proveedores y configuración (el archivo `.env`)

Para configurar un interruptor (embeddings, etiquetado o generación) basta con seleccionar uno de los tres proveedores para cada caso y tener su sección de modelos alimentada con los seleccionados.

> La variable de modelo del etiquetado (la que termina en `TAG_MODEL`) puede rellenerse o no; en caso de no rellenerse, se asume el modelo de generación que esté seleccionado (hereda por omisión).

Las constantes resueltas por el módulo `config.py` a partir del archivo `.env`, con sus valores por omisión (los valores de los modelos son los de la plantilla; para la plantilla canónica comentada ver el archivo `.env.example`):

- **Los interruptores de proveedor** (entre `ollama`, `huggingface` y `google` en todos los casos):
  - `EMBED_PROVIDER` (proveedor de embeddings; valor por omisión, Ollama).
  - `GEN_PROVIDER` (proveedor de generación; valor por omisión, Ollama).
  - `TAG_PROVIDER` (proveedor del etiquetado; vacío por omisión, en cuyo caso hereda el proveedor de generación).
- **Los modelos**, por proveedor e interruptor (las variables `<PROVEEDOR>_EMBED_MODEL`, `<PROVEEDOR>_GEN_MODEL` y `<PROVEEDOR>_TAG_MODEL`, con sus valores por omisión en la plantilla): el modelo de embeddings de cada proveedor, el modelo de generación de cada proveedor y el modelo de etiquetado de cada proveedor (vacío por omisión, en cuyo caso hereda el de generación).
- `CHUNK_SIZE` (la longitud del troceado; valor por omisión, mil) y `CHUNK_OVERLAP` (el solapamiento; valor por omisión, ciento cincuenta).
- `EMBED_DIM` (la dimensión máxima del índice; valor por omisión, 384). La dimensión final del índice es el mínimo entre la dimensión del modelo y esta constante (ver `src/embed.py`).
- `EMBED_BATCH_SIZE` (el lote de los embeddings; valor por omisión, treinta).
- `EMBED_DIM_MAX_OLLAMA`, `EMBED_DIM_MAX_HF` y `EMBED_DIM_MAX_GOOGLE` (la memoria caché offline de la preverificación; vacías por omisión, lo que significa no declarada). Su valor es la dimensión máxima que maneja el modelo cuando la comprobación en línea no es posible.
- `EXPORT_EMBEDDINGS` (activo por omisión): vuelca el archivo `output/embeddings.json` al terminar.
- `TAG_SCORING_DEDUP` (activo por omisión): activa o desactiva el bloque TSD.
- `DEDUP_UMBRAL` (valor por omisión, 0,93): el umbral de deduplicación.
- `CHROMA_DIR` (el directorio del índice; por omisión, el subdirectorio `output/chroma_db`) y `COLLECTION_NAME` (el nombre de la colección; por omisión, `deporte_municipal`).
- Las restantes credenciales y los ajustes de proveedor (la URL de Ollama, el dispositivo y el token de HuggingFace y la clave de la API de Google) se documentan en el apartado de configuración del `[README](../README.md)` y en la plantilla `.env.example`; no se vuelcan en el informe.

## 4. El flujo de invocación (quién llama a quién)

- **El orquestador** (`pipeline.py`, la función `ejecutar_pipeline`) llama, en este orden, a la función de carga del corpus, a la de limpieza, a la de etiquetado, a la de troceado, a la de vectorización, a la de puntuación, a la de deduplicación y a la de indexación, empleando por omisión las constantes del módulo de configuración.
- **El etiquetado** (`tag.py`, la función `etiquetar`) llama al diálogo del proveedor configurado (Ollama, HuggingFace o Google), pasándole los documentos limpios con el metadato de origen poblado.
- **La puntuación** (`scoring.py`, la función `puntuar`) emplea FAISS (el índice plano) y numpy sobre los chunks y los embeddings, y requiere haber corrido antes el etiquetado.
- **La deduplicación** (`dedup.py`, la función `deduplicar`) emplea FAISS (el índice plano) y numpy sobre los chunks y los embeddings, y requiere haber corrido antes la puntuación.
- **La recuperación** (fase pendiente) empleará la función de consulta de la vectorización y el cliente de Chroma, anclados en la constante de nombre de colección.

La dependencia por los **metadatos** es, por tanto: carga, etiquetado, puntuación, deduplicación e indexación. Cada módulo asume que el anterior ya corrió:

- El etiquetado escribe la categoría del documento, los tags, la relevancia y los booleanos de cada tag (por fuente).
- El troceado los propaga a cada chunk junto con el índice de chunk.
- La puntuación lee la relevancia asignada por el modelo y escribe la puntuación semántica.
- La deduplicación lee la puntuación semántica.
- La indexación sana los metadatos (Chroma no admite valores vacíos ni listas).

## 5. Las métricas por consola (en tiempo de ejecución)

- La fase de **carga** registra el número de documentos cargados (e ignorados, en su caso).
- La fase de **limpieza** registra el número de documentos normalizados.
- La fase de **etiquetado** registra, por fuente, la categoría asignada y su relevancia; al final, el número de fuentes etiquetadas, el tiempo empleado y el proveedor y modelo usados.
- La fase de **troceado** registra el número de chunks y la distribución de longitudes (mínimo, percentil veinticinco, media, percentil setenta y cinco, máximo) y el número de chunks cortos (menores de cincuenta caracteres).
- La fase de **vectorización** registra el número de vectores y la dimensión (y, en su caso, un mensaje informativo o de aviso cuando la dimensión declarada difiere de la dimensión máxima del modelo).
- La fase de **puntuación** registra el número de chunks, los scores mínimo, medio y máximo y el mejor chunk (con su puntuación).
- La fase de **deduplicación** registra el umbral usado, el número y el porcentaje de chunks descartados, así como el desglose entre passthrough (política exacta), descartados exactos y semánticos.
- La fase de **índice** registra la colección, la dimensión y el número de vectores insertados frente a los totales de la colección.
- Las **etapas cronometradas** (desde la preverificación hasta la terminación) y la **etapa final** (el tiempo total en segundos, o en milisegundos cuando es inferior a un segundo).

El diccionario devuelto por `ejecutar_pipeline` incluye las mismas métricas para la evaluación.

## 6. El troceado (evaluación en cinco niveles de divisor)

Se ha creado un anexo para saber cómo modificar los niveles de troceado en el proyecto; en él se incluye la escala de niveles, su coste, su beneficio y su indicación, así como el procedimiento completo de validación. Para más información, consultar el [ANEXO del troceado](./ANEXO_CHUNKING.md).

- **El nivel uno (por caracteres fijos)**: método de corte fijo por carácter. No se usa: es rígido e ignora la estructura del texto. Útil solo como experimento de contraste.
- **El nivel dos (recursivo por caracteres)**: es el **actual**. Emplea los separadores de párrafo, frase, espacio y letra; recomendado como punto de partida (el divisor de «cuchilla suiza» de los textos).
- **El nivel tres (específico del documento)**: oportunidad futura. Para los archivos markdown, el divisor específico de markdown (sin añadir dependencias); para las tablas de PDF, la ingesta de tablas con la biblioteca Unstructured (requiere una instalación adicional y un pipeline distinto), una vez el corpus gane en tablas.
- **El nivel cuatro (semántico, por puntos de corte derivados de los embeddings)**: caro (un lote adicional de vectorización por documento fuente). Solo conviene si la evaluación de la recuperación muestra que el nivel dos no basta.
- **El nivel cinco (agéntico, lo decide el modelo de lenguaje)**: no recomendado (lento, caro y no determinista).

**Decisión**: mantener el nivel dos y validarlo con el script de coherencia de los chunks (la «Regla del troceado»: el objetivo es que el dato se recupere con valor, es decir, que el chunk recuperado conserva su sentido al ser recuperado).

## 7. La fase online (recuperación) pendiente

- Se emplearán la función de consulta de la vectorización (`embeddear_consulta`), el cliente de Chroma (`obtener_cliente_chroma`) y el **filtro por la categoría del documento y por los booleanos de cada tag** (la clasificación de la intención mediante palabras clave, con el filtro `where` de Chroma sobre los booleanos que ya están escritos en el índice) y la **reordenación por la puntuación semántica**.
- Con el filtro y la reordenación, se podría subir el parámetro `K` (el número de vecinos recuperados) de tres a cinco sin perder precisión.
