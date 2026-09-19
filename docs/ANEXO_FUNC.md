# ANEXO_FUNC — Funciones de cada archivo

Guía de referencia de las funciones de todos los archivos del proyecto (fuente, scripts, pruebas y datos), destinada a orientar en consultas del tipo «¿dónde está X?». Las firmas corresponden al código actual. La documentación de conceptos (puntuación semántica, deduplicación por política, dimensión garantizada y informes) reside en los demás documentos del [proyecto](../README.md#documentación-técnica).

> Convención: se enumeran las funciones **públicas** (las que acceden al módulo mediante una importación). Las funciones privadas (con prefijo de guion bajo) se mencionarán únicamente cuando sean parte esencial del comportamiento documentado.

## 1. Raíz del proyecto

### `config.py`

Se trata de las constantes del proyecto, leídas del archivo `.env` con valores por defecto. Con ellas se cambia de proveedor sin tocar el código. El módulo no expone funciones de negocio; su interfaz pública es precisamente el conjunto de constantes que se describe más abajo:

- `EMBED_PROVIDER` (valor por defecto, `ollama`): proveedor de embeddings, entre `ollama`, `huggingface` y `google`.
- `EMBED_MODEL` (resolver): el modelo efectivo de embeddings, resuelto según el proveedor; lo emplean tanto el módulo `src/embed.py` como la futura consulta.
- `GEN_PROVIDER` y `GEN_MODEL` (valores por defecto, `ollama` y `llama3.2`): proveedor y modelo de generación para la fase online futura.
- `TAG_PROVIDER` y `TAG_MODEL` (heredan de las constantes de generación): proveedor y modelo del etiquetado; si en el archivo `.env` están vacíos, heredan los valores de generación.
- `OLLAMA_BASE_URL` (valor por defecto, `http://127.0.0.1:11434`): el punto de servicio de Ollama, al que se dirigen las llamadas por lotes a las rutas `/api/embed` y `/api/chat`.
- `HF_DEVICE` (valor por defecto, `cpu`): el dispositivo de los modelos de HuggingFace, con los valores `cpu` y `cuda` (este último exige una tarjeta NVIDIA). La constante `HF_TOKEN` es opcional y solo aplica a los modelos restringidos.
- `GOOGLE_API_KEY` (sin valor por defecto): la clave API; es obligada cuando el proveedor `google` se emplea en alguno de los interruptores.
- `OLLAMA_*_MODEL`, `HF_*_MODEL` y `GOOGLE_*_MODEL` (ver `.env.example`): los modelos por proveedor de cada interruptor (embeddings, generación y etiquetado).
- `CHUNK_SIZE` y `CHUNK_OVERLAP` (valores por defecto, `1000` y `150`): la longitud y el solapamiento del troceado.
- `EMBED_DIM` (valor por defecto, `384`): el tope máximo de dimensión; la dimensión del índice es el mínimo entre la dimensión del modelo y esta constante (ver `src/embed.py`).
- `EMBED_BATCH_SIZE` (valor por defecto, `30`): el tamaño del lote por proveedor.
- `EMBED_DIM_MAX_OLLAMA`, `EMBED_DIM_MAX_HF` y `EMBED_DIM_MAX_GOOGLE` (sin valor declarado por defecto): la memoria caché del modo offline de la preverificación; registran la dimensión máxima que maneja el modelo cuando la comprobación en línea no es posible (red cortada o ausencia de clave API).
- `EXPORT_EMBEDDINGS` (valor por defecto, `true`): exporta el archivo `output/embeddings.json` al terminar el pipeline.
- `TAG_SCORING_DEDUP` (valor por defecto, `true`): activa el bloque TSD completo (etiquetado, puntuación y deduplicación).
- `DEDUP_UMBRAL` (valor por defecto, `0.93`): el umbral de similitud coseno para la deduplicación semántica.
- `CHROMA_DIR` (valor por defecto, `./output/chroma_db`): el directorio persistente del índice.
- `COLLECTION_NAME` (valor por defecto, `deporte_municipal`): el nombre de la colección de Chroma.
- `COSINE_SPACE` (valor por defecto, `cosine`): la métrica de la colección, coherente con los embeddings normalizados.

El módulo contiene una única función interna: el resolver `_resolver` (parámetros: el proveedor, los modelos de lenguaje y el interruptor de proveedor), que resuelve el modelo efectivo de cada interruptor según su proveedor y lanza un error de valor cuando el proveedor no soporta el modelo indicado.

### `__init__.py` (en la raíz)

Por el momento sigue vacío; su único propósito es hacer del directorio un paquete, de manera que el comando `python -m src.pipeline` sea ejecutable desde la raíz del proyecto.

## 2. Paquete `src/` (núcleo del pipeline)

### `src/__init__.py`

La exposición del paquete se rige por **carga perezosa** (mediante la función `__getattr__` y el módulo `importlib`). Se la denomina así porque las dependencias y los submódulos no se cargan al ejecutar `import src`, sino únicamente cuando se accede por primera vez al atributo correspondiente. De este modo no se arrastran de entrada dependencias pesadas (Chroma, sentence-transformers, FAISS): cada submódulo (`load`, `clean`, `chunk`, `embed`, `index`, `pipeline`, `csv_advisor`, `csv_transform`) se importa bajo demanda de la función que precisa de él.

### `src/pipeline.py` — orquestador

Punto de entrada del pipeline completo (preverificación, carga, limpieza, etiquetado, troceado, vectorización, puntuación, deduplicación, indexación e informe), siempre con el interruptor del módulo TSD activado (ver el [README](../README.md)).

Sus símbolos principales son los siguientes:

- `ejecutar_pipeline` (parámetros: rutas, y las opciones por palabra clave `chunk_size`, `chunk_overlap`, `persist_dir`, `collection_name`, `recreate_index`, `export_embeddings` e `informe`); devuelve un diccionario de métricas. Encadena todas las fases, registra el avance por consola y devuelve un diccionario de métricas (documentos, chunks antes y después de la deduplicación, estadísticos de longitud, dimensión, puntuación, deduplicación, índice y tiempos por fase). Genera el archivo `output/informe_index_aaaaMMdd_hhmm.md` salvo cuando se desactiva con `informe=False` en las pruebas.
- El bloque principal `__main__` (mediante la biblioteca argparse): la interfaz de línea de comandos `python -m src.pipeline` con las opciones `[-h]`, `[--chunk-size N]`, `[--chunk-overlap N]`, `[--persist-dir D]`, `[--collection N]`, `[--recreate-index]` y las rutas.
- `_log` (parámetros: fase y mensaje): imprime la marca de tiempo y la fase por consola, con un formato único.
- `_comprobar_dim` (parámetros: dimensión generada, dimensión declarada, y la dimensión máxima declarada del modelo); devuelve una tupla de dos elementos, un nivel y su mensaje. Compara la dimensión generada con `EMBED_DIM` y con la dimensión máxima declarada, y devuelve uno de los siguientes niveles: `vacio` (corpus vacío), `ok` (todo correcto), `info` (la dimensión declarada es menor que la del modelo: se indexa a la declarada) o `aviso` (la dimensión declarada es mayor que la dimensión generada: el índice se crea con la dimensión generada). En ningún caso muta la constante `EMBED_DIM`: las tres dimensiones (declarada/modelo/efectiva) se muestran por separado en el informe.
- `_stats_chunks` (parámetro: los chunks); devuelve un diccionario de estadísticos de longitud: el mínimo, el percentil veinticinco, la media, el percentil cincuenta, el percentil setenta y cinco, el máximo y la cuenta de chunks cortos (menores de cincuenta caracteres, que constituyen ruido probable).

Comportamiento clave:

- El identificador de cada chunk es **global y único**, con el formato `{fuente}::{posición global}`: el índice de chunk por documento es secuencial de cero a n y un archivo CSV produce muchos documentos con la misma fuente; un identificador no global haría que Chroma sobrescribiera filas.
- Si `TAG_SCORING_DEDUP` es `false`, se omiten las fases de etiquetado, puntuación y deduplicación (el pipeline sigue siendo correcto; la deduplicación solo se saltea cuando se desactiva también el bloque TSD).
- El diccionario devuelto contiene las claves que consume el módulo `src/informe.py` (`parametros`, `fases`, `resumen_fases`, `preflight`, `scoring`, `dedup` e `indice`): constituye el contrato entre el pipeline y el informe.

### `src/load.py` — carga del corpus

El módulo abstracta el formato de los archivos: PDF, TXT, MD y CSV se convierten en una lista de documentos (la anotación de tipos `list[Document]`).

Sus símbolos principales son los siguientes:

- `cargar_archivos` (parámetro: una ruta única, una cadena, o una lista de rutas); devuelve una lista de documentos. Es el punto de entrada público del módulo. Admite una ruta única o una lista de archivos y carpetas; recorre estas últimas de forma recursiva (mediante la función `rglob`), aplica el cargador correspondiente a cada extensión y **siempre** garantiza la presencia de los metadatos `source` (nombre base del archivo) y `file_hash`, aunque el cargador no los haya fijado. Los archivos cuya extensión no está soportada se omiten, dejando constancia en consola mediante un aviso.
- `_detectar_encoding` (parámetro: la ruta del archivo); devuelve una cadena. Determina la codificación de caracteres del archivo mediante un intento secuencial de `utf-8-sig`, `cp1252` y `latin-1`; esta última actúa como salvaguarda infalible.
- `_hash_file` (parámetro: la ruta); devuelve una cadena. Calcula el resumen SHA-256 (en notación hexadecimal) de los bytes crudos del archivo, leyéndolo por lotes de un mebibyte (1 MiB) para no cargarlo por completo en memoria.
- Los cargadores por extensión, `_load_pdf`, `_load_text` y `_load_csv`, están registrados en el diccionario `_LOADERS` (una correspondencia de cadenas a funciones de carga): el PDF recurre a `PyPDFLoader`, el TXT y el MD a `TextLoader`, y el CSV delega el tratamiento en el módulo `csv_transform`.

Comportamiento clave: `load.py` es, en rigor, un **lector** exclusivamente. Todo el tratamiento diferencial de los archivos CSV se delega en la función `csv_transform.transform_csv` (cuya decisión la emite el módulo `csv_advisor`).

### `src/clean.py` — normalización de texto

El módulo expone los siguientes procedimientos:

- **`limpiar`** (parámetro: una lista de documentos o una cadena); devuelve el mismo tipo que el argumento recibido. Aplica la función `_normalizar_page_content` a cada documento (o, en su caso, a una cadena aislada, para fines de prueba), devolviendo una copia con los metadatos intactos —una copia superficial del diccionario de metadatos— de modo que los objetos recibidos nunca se alteran.
- **`_normalizar_page_content`** (parámetro: el texto); devuelve una cadena. Aplica, en este orden, las siguientes reglas de normalización: colapsa los saltos de línea múltiples en uno único; colapsa los espacios y tabulaciones múltiples en uno único; elimina las líneas vacías repetidas; recorta el inicio y el final de cada línea; y suprime los caracteres de control y el carácter de orden al inicio (BOM).

### `src/chunk.py` — troceado

- **`trocear`** (parámetros: una lista de documentos o una cadena, y por omisión el tamaño y el solapamiento de chunk); devuelve una lista de documentos. Divide el contenido de cada documento mediante el divisor recursivo de texto por caracteres y constituye el **puente entre la fase de etiquetado y la de troceado** del pipeline. En concreto:
  - Emplea los separadores de párrafo, salto de línea, punto con espacio, espacio y cadena vacía (en este orden: párrafo, luego frase, luego espacio y por último letra), que corresponden al «Nivel 2» recomendado como punto de partida.
  - Admite tanto una lista de documentos como una cadena de texto aislada (útil en pruebas y en evaluaciones).
  - Añade a la metadata de cada chunk las claves `chunk_index` (secuencial **por documento**) y `chunk_size`.
  - Propaga del documento padre a cada chunk las claves `doc_category`, `tags` y `relevancia_llm`, así como las claves propias de la capa CSV.

Comportamiento clave: el módulo respeta la pista de troceado (`chunking_hint`) emitida por el asesor:

- Con la pista `no_chunk`: el documento se conserva íntegro como un único chunk, al constituir una entidad atómica (por ejemplo, una piscina).
- Con la pista `light_chunk`: si su longitud cabe dentro del tamaño de chunk, el documento tampoco se trocea.
- En todo otro caso: se aplica el troceador normal.

La comprobación de valores por omisión se realiza mediante la comparación `is not None`; por consiguiente, un solapamiento de cero es un valor válido que no se interpreta como falsy.

### `src/embed.py` — vectores (multi-proveedor)

Este módulo constituye la única puerta de conversión de texto a vectores del pipeline: tanto la indexación como la consulta futura transitan por él, de modo que la **dimensión se garantiza una única vez**.

Sus símbolos principales son los siguientes:

- **`embeddear`** (parámetros: los textos, y opcionalmente el modelo y la dimensión); devuelve una lista de vectores de valores numéricos. Convierte una secuencia de textos en vectores mediante el proveedor designado por la constante `EMBED_PROVIDER`. Transmite la dimensión como pista al proveedor (el campo `dimensions` del mensaje en Ollama; el parámetro `truncate_dim` en HuggingFace) y aplica, además, la función `_corte_dim` como garantía efectiva: el resultado nunca excede el número de dimensiones indicado (recorte del prefijo seguido de renormalización).
- **`embeddear_consulta`** (parámetro: la pregunta, y opcionalmente el modelo); devuelve un único vector. Calcula el vector de una consulta aislada (fase online futura). Ofrece idéntica garantía de dimensión y emplea el **mismo modelo** que la indexación, condición imprescindible del RAG.
- **`exportar_json`** (parámetros: los chunks, opcionalmente los embeddings, y la ruta, por omisión `output/embeddings.json`): persiste el texto, los metadatos y el vector de cada registro, con fines de inspección y depuración (lo consume el script `scripts/benchmark_tsd.py`).
- **`verificar_modelo_disponible`** (parámetro opcional: los metadatos); devuelve un objeto de resultado de preverificación. Es la **preverificación**: comprueba que el modelo `EMBED_MODEL` esté disponible en el proveedor `EMBED_PROVIDER` **antes** de iniciar el pipeline (el punto `/api/tags` de Ollama, la comprobación `repo_exists` de HuggingFace y el listado de Google, versión v1beta). Si el modelo no está disponible, emite un error de tiempo de ejecución (el pipeline debe interrumpirse); si la verificación en línea no es posible (red cortada, Ollama apagado o ausencia de clave API), recurre a la memoria caché del archivo `.env` —la constante `EMBED_DIM_MAX_` del proveedor activo— y devuelve el indicador `verificado_por` con el valor `cache_env`, junto con un aviso.
- **Las funciones de vectorización por proveedor**, `_embed_ollama`, `_embed_huggingface` y `_embed_google`: las implementaciones particulares, cada una con sus parámetros de textos. Lotización en Ollama por lotes de tamaño `EMBED_BATCH_SIZE` (la ruta `/api/embed`); en HuggingFace, el cargador `SentenceTransformer` con memoria caché y la normalización de embeddings activada; y la API de servicio web `batchEmbedContents` en Google.
- **Los auxiliares vectoriales**, `_normalizar_vector` (o su equivalente `_normalizar`) y `_corte_dim` (parámetros: los vectores y el tope de dimensión): la renormalización por la norma euclídea, con salvaguarda contra la división por cero, y el recorte del prefijo al tope indicado seguido de renormalización, en coherencia con la métrica coseno de Chroma.

Comportamiento clave: el registro `_PROVIDERS` mantiene la correspondencia entre proveedores y funciones; añadir un proveedor nuevo consiste únicamente en agregar una función y una línea al diccionario.

### `src/index.py` — ChromaDB

El módulo expone los siguientes símbolos:

- `obtener_cliente_chroma` (parámetro opcional: el directorio persistente); devuelve una instancia del cliente de Chroma. Devuelve un cliente persistente anclado en `CHROMA_DIR`, concebido para su reutilización en la futura fase de recuperación.
- `obtener_guid_chroma` (parámetro opcional: el directorio persistente); devuelve una cadena o vacío. Devuelve el identificador único de la carpeta de datos de ChromaDB situada en el directorio indicado, o bien un valor vacío cuando dicha carpeta no existe. El módulo `pipeline.py` lo emplea para nombrar el informe (prefijo de ocho caracteres del identificador) y para detectar la existencia de un índice previo.
- `crear_coleccion` (parámetros: el cliente y opcionalmente el nombre): crea o recupera la colección con la métrica coseno, coherente con los embeddings normalizados.
- `indexar` (parámetros: los identificadores, los embeddings, los documentos, los metadatos, y opcionalmente el directorio persistente, el nombre de la colección y la opción de recreación); devuelve una tupla con los identificadores vivos y el total de la colección. Inserta en lotes (la constante de lote es `2000`; el límite efectivo del backend varía con la dimensión), sanea los metadatos —Chroma no admite valores vacíos ni listas, de modo que los convierte en la cadena `null` y en cadenas, respectivamente—, admite la opción de recreación y **verifica** la inserción (lote a lote, contra la consulta del propio Chroma) antes de devolver la cifra. Puesto que el total proviene del recuento de la colección, una reindexación parcial resulta visible en el resultado.
- `_sanear_meta` (parámetro: los metadatos): sanea los metadatos previos a su persistencia en Chroma.

Comportamiento clave: la verificación se realiza lote a lote, pues una consulta masiva de identificadores haría estallar a Chroma con el error de variables SQL excesivas; asimismo, la comprobación mediante aserción sobre identificadores únicos protege contra la duplicación (ver `src/pipeline.py`).

### `src/informe.py` — informe de indexación

Este módulo se limita a renderizar en markdown el diccionario de métricas del pipeline (por defecto, se escribe en `output/informe_index_aaaaMMdd_hhmm.md`); no lee configuración ni produce efectos secundarios, de modo que puede probarse de forma aislada.

Sus símbolos principales son los siguientes:

- `generar_informe` (parámetros: los datos del pipeline y opcionalmente la ruta); devuelve la ruta del archivo escrito. Escribe el informe —compuesto por siete secciones: parámetros, preverificación, métricas por fase, troceado, índice, bloque TSD y señales— en la ruta indicada (por omisión, `output/informe_index_aaaaMMdd_hhmm.md`, con fecha y hora locales) y devuelve la ruta escrita.
- `formatear_duracion` (parámetro: los segundos); devuelve una cadena. Formatea la duración de forma legible: muestra un guion largo para un valor vacío o para valores inferiores a medio centésima de segundo (evitando falsos ceros), `X ms` cuando la duración es inferior a un segundo, y `X.XX s` en el resto de los casos.
- Los bloques internos de renderizado, `_tabla` (parámetro: las filas), `_fases_md` (parámetros: las fases y el resumen), `_scoring_md` (parámetros: la puntuación y la deduplicación) y `_senales` (parámetro: los datos): corresponden a las secciones de parámetros, de fases, del bloque TSD y de las señales automáticas de decisión.

Las condiciones de las señales (condición y acción recomendada) y las reglas de la tabla de parámetros están documentadas en [`ANEXO_OUTPUTS.md`](./ANEXO_OUTPUTS.md).

## 3. Capa CSV (clasificación diferencial)

### `src/csv_advisor.py`

Este módulo determina el **tratamiento** que corresponde a cada archivo CSV antes de su troceado. Para ello recurre a un esquema de decisión en dos niveles, dispuesto de lo más económico a lo más costoso y que, en ambos casos, no recurre a embeddings. Sus símbolos principales son los siguientes:

- **`advise_csv`** (parámetro: la ruta del archivo); devuelve una tupla con el consejo y el perfil. Constituye el punto de entrada del módulo. Resuelve la decisión aplicando, en primer término, la receta de la fuente conocida (la constante `RECETAS_CONOCIDAS`, con una confianza de 0.99) y, cuando el esquema ha cambiado o la fuente resulta desconocida, la heurística del perfil estructural. En ningún caso devuelve un valor vacío.
- **`inspect_csv`** (parámetros: la ruta y opcionalmente el tamaño de muestra, por omisión dos mil filas); devuelve el perfil de la estructura. Elabora el perfil estructural sobre una muestra de filas: identifica las columnas de identificación, de medida, de carácter temporal y de contenido narrativo —a partir de los tokens del nombre de cada columna—, así como la densidad numérica real de cada una, las cardinalidades y el índice de unicidad de las filas.
- **`match_known_source`** (parámetro: el perfil); devuelve un consejo o vacío. Aplica la receta fija de las fuentes conocidas de datos.madrid.es; devuelve un valor vacío cuando la fuente no es conocida o cuando ya no contiene las columnas esperadas, caso en el que la decisión se delega en la heurística.
- **`infer_csv_kind`** (parámetro: el perfil); devuelve el consejo. Aplica la heurística del perfil, que resuelve la decisión en este orden: primer lugar, entidad (identificador estable y ausencia de dimensión temporal), lo que produce el tratamiento de documento de entidad con deduplicación por clave exacta; segundo lugar, hecho o serie temporal (medidas numéricas y al menos tres dimensiones), lo que produce el tratamiento de documento agrupado con deduplicación por grupo; tercer lugar, contenido textual, lo que produce el tratamiento de fila como documento con deduplicación semántica opcional; y cuarto lugar, como refugio conservador, el archivo desconocido con deduplicación estrictamente por clave.
- **`leer_filas_csv`** (parámetros: la ruta y opcionalmente el tamaño de muestra); devuelve una tupla con las columnas y las filas. Proporciona la lectura robusta y compartida del corpus: detecta la codificación y el delimitador (punto y coma, coma o tabulación), normaliza las claves a su forma basificada (por ejemplo, la cabecera `MXASSETNUM,C,12` queda reducida a `mxassetnum`) y corrige el ordinal del corpus, que aparece corrupto según la codificación empleada.
- **Las utilidades auxiliares**: `detectar_encoding`, `normalizar_col` y `detectar_delimitador` (cada una de sus parámetros). Reúnen las funciones auxiliares de lectura; el delimitador se resuelve como aquel que produce un número estable de columnas entre la cabecera y los datos.

En cuanto a los tipos, el perfil (`CsvProfile`) recoge las métricas estructurales del archivo, mientras que el consejo (`CsvAdvice`) materializa la decisión ejecutable, con los campos `csv_kind` (la clase del archivo), `treatment` (el tratamiento), `dedup_policy` (la política de deduplicación), `id_columns` (las columnas identificador), `measure_columns` (las columnas de medida), `grouping_keys` (las claves de agrupación), `confidence` (la confianza en la decisión) y `reason` (la justificación).

**La constante `RECETAS_CONOCIDAS`** (las fuentes conocidas de datos.madrid.es) no se genera automáticamente: se trata de un diccionario **curado a mano** a partir del perfilado del corpus real (mediante `inspect_csv`) y se conserva en el código (la clave es el nombre base del origen y la confianza, fijada en 0.99). Cada entrada de este diccionario es, a su vez, una receta compuesta por los siguientes elementos:

- El triplete de decisión (`kind`, `treatment` y `dedup`): es la clave que el pipeline aplica. Por ejemplo, una tabla de entidad produce el tratamiento de documento de entidad con deduplicación por clave exacta, o una tabla de hechos produce el tratamiento de documento agrupado con deduplicación por grupo.
- Las columnas de identificador estable: `pk`, `mxassetnum`, `id-evento`, entre otras. Resultan ausentes en las tablas de hechos, pues sus filas no son entidades.
- Las claves de agrupación (campo opcional, exclusivo de las tablas de hechos): las dimensiones por las que se agrupan las filas (por ejemplo, un triplete de mes, centro deportivo y tipo de abono).
- Las columnas de medida (campo opcional, exclusivo de las tablas de hechos): las columnas numéricas que el conjunto de datos debe aportar.
- La justificación (`reason`): es la que muestra el registro `[CSV]` por consola.

Los nombres de columna figuran en **formato basificado** (el mismo que el de las claves de la función de lectura de filas: letras minúsculas y sin sufijo técnico). La función `match_known_source` exige que todas esas columnas se sigan presentando; si el conjunto de datos se regenera bajo un esquema distinto, devuelve un valor vacío y la heurística `infer_csv_kind` asume el relevo —degradando la confianza desde 0.99 hasta un rango de 0.4 a 0.85— sin comprometer el funcionamiento del pipeline. Para extender las recetas con una fuente nueva o para corregirlas, procede perfilar el archivo con `inspect_csv`, revisar las columnas, los identificadores y las medidas, y añadir (o corregir) la entrada correspondiente. Las entradas actuales las fija la prueba parametrizada `tests/test_csv_layers.py` en su clase de asesor (TestAdvisor).

La decisión que se aplica al corpus actual (el directorio `data/`), ya sea por receta conocida o por heurística, es la siguiente:

- El archivo de polideportivos: la columna identificadora es `pk`, la clasificación es tabla de entidad y el tratamiento es documento de entidad con deduplicación por clave exacta.
- El archivo de instalaciones deportivas: la columna identificadora es `pk`, la clasificación es tabla de entidad y el tratamiento es documento de entidad con deduplicación por clave exacta.
- El archivo de piscinas públicas: la columna identificadora es `pk`, la clasificación es tabla de entidad y el tratamiento es documento de entidad con deduplicación por clave exacta.
- El archivo de áreas deportivas: la columna identificadora es `mxassetnum`, la clasificación es tabla de entidad y el tratamiento es documento de entidad con deduplicación por clave exacta.
- El archivo de agenda de actividades deportivas: la columna identificadora es `id-evento`, la clasificación es tabla de entidad y el tratamiento es documento de entidad con deduplicación por clave exacta.
- El archivo de abonos en deportes: sin columna identificadora, la clasificación es tabla de hechos y el tratamiento es documento agrupado con deduplicación por grupo.
- El archivo de descuentos en deportes: sin columna identificadora, la clasificación es tabla de hechos y el tratamiento es documento agrupado con deduplicación por grupo.

Estas decisiones se fundan en la distinción entre dos clases de tablas:

- **Las tablas de entidad** (centros, instalaciones, piscinas, áreas y eventos): cada fila representa una entidad dotada de **identificador estable**. Por ello se documenta cada fila como entidad (tratamiento de documento de entidad) y se deduplican **por clave exacta**. Se evita la deduplicación semántica porque dos filas casi idénticas (por ejemplo, dos piscinas del mismo barrio) constituyen entidades distintas, y no duplicados.
- **Las tablas de hechos** (abonados y descuentos): carecen de columna identificadora y cada fila es un registro repetitivo. Por consiguiente, se agrupan por sus dimensiones (tratamiento de documento agrupado) y únicamente se deduplican los grupos repetidos, de modo que las filas no compiten entre sí en la deduplicación semántica.

**Glosario de los registros**:

- **Los términos «PK estable» e «identificador estable»**: ambas denominaciones designan el mismo concepto dentro de la constante `RECETAS_CONOCIDAS`; la diferencia es puramente cosmética. Se emplea el término «PK» cuando la columna se denomina literalmente `pk`, y se emplea «identificador estable» cuando el identificador lleva otro nombre (por ejemplo, `mxassetnum` o `id-evento`). No se trata, en consecuencia, de un concepto distinto.
- **El término «índice»**: únicamente existe el **índice vectorial de Chroma** (el directorio `CHROMA_DIR`); no existe, propiamente dicho, un «índice estable» como tal. Si se modifica el modelo de embeddings o la política de troceado, el índice se regenera mediante el flag `--recreate-index`.

### `src/csv_transform.py`

Este módulo materializa el consejo emitido por `csv_advisor`: convierte las filas del archivo CSV en documentos enriquecidos. Sus símbolos principales son los siguientes:

- `transform_csv` (parámetro: la ruta del archivo); devuelve una lista de documentos. Es el punto de entrada, invocado por el módulo de carga. Emite el registro `[CSV]` que documenta la decisión adoptada (clase, tratamiento, deduplicación y confianza) y aplica la estrategia correspondiente al tratamiento determinado.
- `entidad_a_documento` (parámetros: la fuente, la fila, la posición de la fila, las columnas y el consejo): convierte una fila de entidad en un único documento, dotado de la política de deduplicación por clave exacta, de un identificador de entidad derivado de la columna de ID y de la pista de troceado sin troceado; incorpora además las columnas de distrito y de barrio cuando tales columnas existen.
- `grupo_a_documento` (parámetros: la fuente, las claves de agrupación, las filas, las columnas y el consejo): agrupa las filas de un hecho según sus claves de agrupación y las materializa como un único documento agregado, con su desglose línea a línea, bajo la política de deduplicación por grupo, con la marca de agregado activada y la pista de troceado ligero.
- `filas_a_documentos` (parámetros: la fuente, las filas, las columnas y el consejo): materializa cada fila como un documento plano, destinado a tablas textuales o a los casos de archivo desconocido.
- `limpiar_valor` (parámetro: el valor de la celda); devuelve una cadena. Compacta los espacios consecutivos de un valor; se emplea durante la materialización de las filas.

Comportamiento clave: la metadata (la política de deduplicación, la clave de entidad, la clave de grupo y la pista de troceado) acompaña a cada chunk hasta el índice y gobierna con qué política se deduplica ese chunk; la deduplicación de los archivos CSV **nunca es semántica**, sino estrictamente por clave exacta.

## 4. Paquete `src/tsd/` (etiquetado, puntuación, deduplicación)

Este paquete encadena el etiquetado semántico, la puntuación y la deduplicación, de modo que al índice entre únicamente señal limpia. Cada una de sus tres etapas escribe en la metadata que la etapa siguiente consumirá (de la fase de etiquetado a la de troceado, de ahí a la de puntuación y de esta a la de deduplicación).

### `src/tsd/tag.py` — etiquetado con modelo de lenguaje

- `etiquetar` (parámetro: los documentos); devuelve una lista de documentos. Interroga al modelo de lenguaje **una única vez por fuente** (no por página ni por fila) y propaga a todos los documentos de esa fuente los campos de categoría del documento, los tags (una cadena unida por punto y coma, dado que Chroma no admite listas; su formato es de lectura humana), la relevancia calculada por el LLM (un valor entre cero y uno) y los **booleanos por tag** que toman valor verdadero cuando el tag está presente (escritura escasa). La taxonomía es cerrada: las categorías son tarifas, normativa, reservas, abonos, instalaciones y agenda; los dieciséis tags son abono, piscina, reserva, tarifa, horario, empadronado, descuento, competición, instalación, precio, cancelación, devolución, accesibilidad, aire libre, inscripción y temporada. Dichos booleanos constituyen el contrato de filtrado de la fase online (el filtro de Chroma). Si el LLM devuelve un objeto JSON malformado, se aplica un etiquetado por defecto (categoría instalaciones y relevancia 0.5) en lugar de interrumpir el pipeline.
- `CATEGORIAS_VALIDAS` y `TAGS_VALIDAS`: son conjuntos inmutables que fijan la taxonomía cerrada; el analizador filtra los valores devueltos contra ellos (una categoría fuera de la taxonomía se sustituye por la categoría de instalaciones, con aviso en consola, y los tags fuera se descartan silenciosamente), garantizando que la metadata del índice permanece filtrable en la fase online.
- Los diálogos por proveedor, `_chat_ollama`, `_chat_huggingface` y `_chat_google` (cada uno con sus parámetros de proveedor, modelo, mensajes y opciones): son las funciones de diálogo por proveedor (Ollama mediante la ruta `/api/chat`, HuggingFace mediante el proceso de generación de texto con memoria caché, y Google mediante la API de servicio web para generación de contenido), registradas en el diccionario del chat (añadir un proveedor supone una función adicional y una línea de registro).
- `_parse_json` (parámetro: el texto de la respuesta); devuelve un diccionario. Extrae el objeto JSON de la respuesta del modelo de lenguaje, tolerando los bloques de marcado de JSON y la prosa previa o posterior; lanza un error de decodificación si no se localiza ningún objeto válido.

Nota de coste: el mensaje de indicación (el prompt) únicamente incorpora los **primeros 6000 caracteres** de cada fuente; se trata de un compromiso entre el coste y la fiabilidad que asegura que cada fuente queda etiquetada de forma consistente. La duración de la llamada y el proveedor y modelo empleados se registran en consola.

Nota de filtrado (fase online): los tags se conservan como una cadena unida por punto y coma, de lectura humana, que **no** admite filtrado por subcadena mediante el filtro de Chroma —únicamente admite la coincidencia exacta de la cadena completa—. El filtrado efectivo lo ejercen los **booleanos por tag** que la función de etiquetado escribe adicionalmente: la clave del tag con el valor verdadero, escrita de forma escasa (únicamente se crea la clave del tag presente, por ejemplo, la clave de tag de piscina o la de tag de cancelación). Dado que el filtro de Chroma soporta los booleanos y una clave ausente no produce coincidencia, la fase online (el flag de consulta del mapa de ruta) filtra directamente sobre el índice:

```python
# filtrado simple
where={"tag_piscina": True}
# filtrado compuesto: categoría + tag
where={"$and": [{"doc_category": "reservas"}, {"tag_cancelación": True}]}
```

Los booleanos se propagan junto con la metadata (de la fase de etiquetado a la de troceado) y sobreviven al saneo que aplica `index.py` (Chroma acepta los valores verdadero y falso), de modo que cada chunk del índice resulta filtrable por categoría, por tag y por su puntuación semántica (mediante la reordenación).

### `src/tsd/scoring.py` — puntuación semántica

- `puntuar` (parámetros: los chunks, los embeddings y opcionalmente una referencia para devolver las métricas); devuelve una lista de documentos. Calcula la puntuación semántica de cada chunk mediante la suma ponderada —un factor de 0.45 sobre la relevancia asignada por el LLM, un factor de 0.20 sobre la centralidad, un factor de 0.20 sobre la no redundancia (es decir, sobre uno menos la redundancia) y un factor de 0.15 sobre la autoridad—, recortada al intervalo comprendido entre cero y uno y redondeada a cuatro decimales, y la guarda en el metadato `semantic_score`. Asimismo vuelca un conjunto de métricas en la referencia indicada: el valor mínimo, medio y máximo, la centralidad media, la redundancia media, el porcentaje de chunks por encima de 0.6 y el tiempo de ejecución.
- `_autoridad` (parámetro: la fuente); devuelve un valor numérico. Asigna un peso según el tipo de fuente, mediante coincidencia por subcadena sobre el nombre de la fuente: reglamento o normativa, valor 1.0; precios o tarifas, valor 0.9; agenda, valor 0.6; y el resto, valor 0.7.
- `_redundancias` (parámetro: la matriz de embeddings); devuelve un vector de redundancias numéricos. Calcula la similitud coseno de cada vector respecto a su vecino más cercano, excluido el propio, apoyándose en FAISS (el índice plano de producto interno) con una búsqueda de dos vecinos. Con ello reduce el consumo de memoria desde uno proporcional al cuadrado del número de filas, hasta uno proporcional a las filas por la dimensión: aproximadamente 1,3 gigabytes frente a los 68 gigabytes de la matriz completa, en la escala real del proyecto (cerca de 130 mil filas y 2560 dimensiones). La búsqueda de dos vecinos es suficiente porque la mejor similitud de cada fila figura siempre entre sus dos primeros vecinos.

### `src/tsd/dedup.py` — deduplicación

- `deduplicar` (parámetros: los chunks, los embeddings, el umbral por omisión y opcionalmente una referencia para devolver las métricas); devuelve una tupla con los chunks conservados y sus embeddings. Ejecuta una deduplicación **consciente de política**. Los chunks con una política de deduplicación estricta (por clave de entidad, por grupo o estrictamente por clave) se deduplican **únicamente por su clave** —es decir, ante la repetición literal— y **nunca por similitud coseno**; el resto pasa por la deduplicación semántica por recorte greedy, ordenada por la puntuación semántica en orden descendente. Vuelca las métricas en la referencia indicada: el umbral, el recuento previo y posterior, los descartados exactos frente a los semánticos, el tiempo, la cobertura por las seis categorías cerradas (previa y posterior) y los tres tags más frecuentes del posterior.
- `_deduplicar_semantico` (parámetros: los chunks, los embeddings y el umbral); devuelve una tupla con las posiciones conservadas y los embeddings conservados. Aplica una estrategia de recorte greedy e incremental sobre FAISS: cada candidato se busca (un vecino) contra el índice plano que va creciendo a medida que se conservan chunks. Su semántica es idéntica a la de la matriz completa de similitud (descartar cuando el cociente es mayor o igual que el umbral), pero con un consumo de memoria lineal.
- `_clave_exacta` (parámetro: el chunk); devuelve una tupla. Construye la clave de identidad según la política aplicable: por entidad (la fuente más la clave de entidad), por grupo (la fuente más la clave de grupo y la posición del chunk), o por fila literal (la fuente más la fila y su contenido).

## 5. Scripts de validación y experimentación

### `scripts/eval_coherencia_chunks.py`

Evalúa la coherencia de los cortes aplicados al texto. Para ello, trocea un texto de dominio repetido empleando los parámetros de tamaño y solapamiento definidos en el archivo `.env`, genera los embeddings con el proveedor configurado y compara la similitud coseno media entre chunks adyacentes frente a la media de pares aleatorios. Un margen superior a 0.05 indica que el solapamiento preserva la coherencia temática del documento; por el contrario, un valor próximo a cero señala que los cortes resultan arbitrarios. El script se ejecuta mediante el comando `python -m scripts.eval_coherencia_chunks`.

### `scripts/benchmark_tsd.py`

Verifica, a escala real, el coste de las fases del bloque TSD (puntuación y deduplicación) tras la migración a FAISS. Para ello, carga el archivo `output/embeddings.json` —lo cual exige que la constante `EXPORT_EMBEDDINGS` esté activada— y ejecuta las funciones de puntuación y de deduplicación sin depender de ChromaDB ni de Ollama. El script se ejecuta mediante el comando `python -m scripts.benchmark_tsd`.

### `scripts/test_embeddings.py`

Realiza una prueba rápida de los modelos de embedding disponibles en Ollama. Para ello, emplea cuatro frases (tres relacionadas entre sí y una irrelevante), calcula la similitud coseno por pares y toma el **margen** (la diferencia entre la similitud relevante y la irrelevante) como indicador del poder de separación del dominio. Resulta especialmente útil para elegir el valor de la constante de modelo de embeddings de Ollama. Requiere que Ollama esté en marcha y que los modelos estén descargados. El script se ejecuta mediante el comando `python -m scripts.test_embeddings`.

## 6. Pruebas (el directorio `tests/`)

Suite de pruebas de pytest que funciona offline (proveedores falsos o simulados; sin red y sin efecto sobre el directorio de salidas):

- `test_load.py`: verifica la detección de la codificación (UTF-8, CP1252 y Latin-1, con carácter de orden al inicio y bytes corruptos) y los cargadores TXT y CSV; comprueba que la carga se completa sin errores ni carácter fantasma.
- `test_clean.py`: valida la función de limpieza frente al carácter de orden al inicio, al colapso de espacios y líneas vacías, a la eliminación de caracteres de control, a la preservación del contenido y a las métricas de compresión.
- `test_chunks.py`: comprueba la función de troceado en cuanto a los límites de tamaño, el solapamiento entre adyacentes, los metadatos (el índice de chunk y la fuente) y la preservación de palabras y números (las tarifas y los horarios); además verifica las métricas de troceado destinadas al informe.
- `test_embed.py`: cubre la preverificación (Ollama, HuggingFace, Google y la memoria caché offline de las dimensiones máximas), el recorte de la dimensión y la seguridad dimensional (las funciones de vectorizar y de comprobación de dimensión).
- `test_csv_layers.py`: ejercita la capa CSV sobre las familias reales del corpus: perfilado (codificación, cabeceras, medidas), recetas conocidas frente a la heurística, transformación (entidad, grupo y fila) y deduplicación por la política de deduplicación.
- `test_tsd.py`: valida la puntuación (el valor total, la penalización por redundancia, la autoridad normativa y la acotación a 1.0) y la deduplicación (la conservación del chunk de mejor puntuación, los umbrales, el orden original, el volcado de cobertura por categoría y los tres tags principales); incluye, además, una verificación contra una referencia de matriz completa (300 vectores).
- `test_informe.py`: comprueba la generación del informe en el renderizado (parámetros, fases, troceado, bloque TSD y la cobertura por categoría de la sección 6.3) y que las funciones de puntuar y deduplicar vuelcan sus métricas en el diccionario de métricas.
- `test_informe_e2e.py`: la prueba de extremo a extremo offline con proveedor falso: la ejecución completa del pipeline, el diccionario de métricas completo, el informe con sus siete secciones y ChromaDB con los vectores esperados.
- `test_preguntas.py`: valida el archivo de preguntas (una instantánea estable: 18 preguntas, con orden e identificadores correctos), que la categoría esperada pertenezca a la taxonomía cerrada y que los tags esperados pertenezcan a los tags válidos (incluida la pregunta multi-aspecto, la número ocho).

La suite se ejecuta mediante el comando `python -m pytest tests/ -v`.

## 7. Otros (archivos de apoyo)

- **El archivo `queries/preguntas.json`**: es el conjunto de 18 preguntas de evaluación del dominio (tarifas, abonos, reservas y agenda). Cada pregunta es un objeto JSON con cinco campos: el identificador (un número entre 1 y 18), la pregunta literal, la categoría esperada (la categoría primaria única, perteneciente a la taxonomía cerrada), la fuente esperada (la fuente del corpus que debería dar el acierto) y los tags esperados (una lista de tags perteneciente a los tags válidos, multi-aspecto: refleja los tags que el documento debería llevar, a fin de evaluar la coincidencia de etiquetas en la fase online).
- **El directorio `data/`**: el corpus, compuesto por ocho PDF de normativa y tarifas, siete CSV de datos.madrid.es y un TXT de agenda. Los CSV pueden regenerarse descargando cualquiera de los conjuntos de datos del [grupo de deporte de datos.madrid.es](https://datos.madrid.es/group/deporte); el cargador los recarga igual aunque el esquema cambie levemente.
- **El directorio `output/`**: los artefactos generados (ignorados por git): el subdirectorio de Chroma (el índice persistente), el archivo de embeddings (cuando la constante de exportación está activada) y el informe (uno por ejecución).
- **El archivo `.env`**: la configuración local (ignorado por git); la plantilla canónica es el archivo `.env.example`.
- **El archivo `config.py`**: las constantes del proyecto, ya documentadas en la sección primero.
