# Generación Aumentada por Recuperación (RAG) para el Deporte Municipal del Ayuntamiento de Madrid

Este es un sistema **RAG** (por sus siglas en inglés, Retrieval Augmented Generation, o generación aumentada por recuperación) para responder preguntas mediante la recuperación de contenido sobre los documentos de **deporte municipal de Madrid**: tarifas, abonos, reservas, horarios e instalaciones.

La carga **incluye** un corpus real de documentos (PDF, CSV y TXT) de la sede de datos abiertos del Ayuntamiento de Madrid. Posteriormente, el sistema trocea el corpus, genera los embeddings, evalúa y mejora la calidad semántica mediante etiquetado con un modelo de lenguaje, puntuación y deduplicación (el bloque TSD, que se describe más adelante, es opcional), y guarda el resultado en la base de datos vectorial **ChromaDB** para la recuperación semántica. Se trata de un pipeline **offline** completo; la fase online (recuperación, generación e interfaz de usuario) consta en el [mapa de ruta](#roadmap).

El pipeline encadena las fases en este orden:

```text
PREFLIGHT → LOAD → CLEAN → TAG (LLM) → CHUNK → EMBED → SCORING → DEDUP → INDEX → INFORME
```

## Características principales

- **Preverificación de disponibilidad**: el pipeline comprueba que el modelo `EMBED_MODEL` está disponible en el proveedor `EMBED_PROVIDER` (el punto de servicio `/api/tags` de Ollama, la comprobación `repo_exists` de HuggingFace o el listado de modelos de Google) y **se detiene** si no lo está. Cuando no es posible verificarlo en línea (red cortada o ausencia de clave API), degrada la comprobación a la memoria caché offline (`EMBED_DIM_MAX_*`) con un aviso, en lugar de romper el proceso.
- **Proveedor múltiple** sin tocar el código: se elige entre `ollama`, `huggingface` y `google` mediante variables de entorno, con **tres interruptores independientes** por proveedor: `EMBED_PROVIDER`, `TAG_PROVIDER` y `GEN_PROVIDER`. El etiquetado hereda el proveedor de generación por defecto cuando no se define, en todos los casos dentro de los tres anteriores.
- **Bloque TSD** (etiquetado, puntuación y deduplicación, opcional): primero etiqueta cada fuente con un modelo de lenguaje mediante una taxonomía cerrada; después asigna a cada chunk una puntuación semántica que combina la relevancia del LLM con la centralidad y la no redundancia (ambas medidas sobre los vectores con FAISS) y la autoridad (un ajuste fino según el tipo de fuente); y por último aplica una deduplicación *greedy*: ordena los chunks por puntuación y conserva cada uno únicamente cuando no es demasiado parecido a los que ya se han conservado (el umbral lo fija `DEDUP_UMBRAL`). El proceso respeta la política específica de los archivos CSV, de modo que el índice contiene más información útil y menos contenido redundante.
- **Tratamiento diferenciado de los archivos CSV**: el módulo `csv_advisor` analiza y clasifica cada archivo (entidad con identificador estable, tabla de hechos, textual o desconocido) y decide cómo convertirlo en documento para indexación y cómo aplicar la deduplicación. La deduplicación de los archivos CSV **nunca es semántica**; se realiza por clave exacta. Ver el [detalle](docs/ANEXO_FUNC.md#3-capa-csv-clasificación-diferencial).
- **Dimensión garantizada**: la dimensión final de todos los vectores del índice es siempre el mínimo entre la dimensión generada y la constante `EMBED_DIM`. Cuando el proveedor lo permite, se solicita esa dimensión en el servidor (en Ollama mediante el parámetro `dimensions` y en HuggingFace mediante `truncate_dim`); después, el cliente recorta el prefijo del vector si es necesario y lo renormaliza. Si `EMBED_DIM` es menor que la dimensión del modelo, se emite un mensaje informativo (tipo INFO); si es mayor que la dimensión generada, se muestra un aviso (tipo AVISO) y se usa la dimensión generada —el mensaje deja constancia de ambas dimensiones—, **evitando así incompatibilidades en ChromaDB**.
- **Índice regenerable** en ChromaDB persistente (métrica coseno): cada ejecución puede reconstruir el índice a partir del corpus, sin depender de inserciones anteriores. Los vectores se insertan por lotes para controlar el uso de memoria y mejorar el rendimiento. Al finalizar, el pipeline vuelve a consultar los identificadores insertados y compara el resultado con lo esperado, de modo que detecta inserciones incompletas o reindexaciones parciales antes de dar el proceso por válido.
- **Informe de indexación** en formato markdown al terminar cada ejecución (archivo `output/informe_index_<GUID8>_YYYYMMDD_HHMM.md`, en el que `<GUID8>` son los ocho primeros caracteres del identificador único de ChromaDB): parámetros aplicados, resultados de la preverificación, tiempos por fase, troceado, índice, bloque TSD y **señales automáticas** con la acción recomendada en cada caso.
- **Registro en consola** de cada fase, además del diccionario de métricas que devuelve la función `ejecutar_pipeline`, listo para su evaluación.

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
│   ├── pipeline.py        # orquestador: encadena todas las fases + dict de métricas
│   ├── load.py            # LOAD: PDF/TXT/MD/CSV → Documents (archivo o carpeta)
│   ├── clean.py           # CLEAN: normaliza saltos, espacios, BOM, control chars
│   ├── chunk.py           # CHUNK: RecursiveCharacterTextSplitter (Level 2)
│   ├── embed.py           # EMBED: multi-proveedor (ollama / huggingface / google)
│   ├── index.py           # INDEX: ChromaDB persistente, métrica coseno, verificado
│   ├── informe.py         # renderiza output/informe_index_<GUID8>_YYYYMMDD_HHMM.md
│   ├── csv_advisor.py     # clasifica cada CSV y decide tratamiento + dedup_policy
│   ├── csv_transform.py   # materializa el consejo: convierte las filas en Documents
│   └── tsd/
│       ├── tag.py         # TAG: etiquetado LLM por fuente (taxonomía cerrada)
│       └── scoring.py     # SCORING: relevancia + centralidad + redundancia + autoridad
│       └── dedup.py       # DEDUP: greedy por similitud coseno (FAISS)
├── data/                   # corpus: 8 PDF + 7 CSV + 1 TXT
├── queries/                # preguntas de evaluación (18)
├── scripts/                # validación de chunking, embeddings y benchmark TSD
├── tests/                  # pruebas pytest offline (load/clean/chunk/embed/csv/tsd/informe)
├── docs/                   # documentación técnica (incluye ANEXO_FUNC)
└── output/                 # índice Chroma, embeddings.json, informe (gitignored)
```

Para conocer las funciones de cada archivo (firmas y comportamiento), consultar [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md).

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/dcruzpuri/project-rag-hmd-deporte-municipal.git
cd project-rag-hmd-deporte-municipal
```

### 2. Crear el entorno virtual e instalar las dependencias

En Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS o Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar el archivo `.env`

```bash
cp .env.example .env
```

Se elige el proveedor en cada interruptor (`EMBED_PROVIDER`, `TAG_PROVIDER` y `GEN_PROVIDER`, cuyo valor puede ser `ollama`, `huggingface` o `google`):

- **Ollama**: arrancar Ollama con el modelo de embeddings y el modelo de lenguaje descargados (la dirección base se configura en `OLLAMA_BASE_URL`, cuyo valor por defecto es `http://127.0.0.1:11434`).
- **HuggingFace**: configurar `HF_DEVICE` (el valor `cpu` o `cuda`) y, únicamente si el modelo es de acceso restringido, introducir el token propio en `HF_TOKEN`.
- **Google**: introducir la clave propia en `GOOGLE_API_KEY`.

> **Nota:** si se cambia el modelo de embeddings o el troceado, es preciso **regenerar el índice** con el flag `--recreate-index` (ver la sección [Uso](#uso)).

## Uso

### Indexar el corpus

Desde la raíz del proyecto:

```bash
python -m src.pipeline data --recreate-index
```

La consola muestra el avance fase a fase (las líneas sin marca de tiempo son el detalle que emiten los módulos internos, por ejemplo `[TAG]   <fuente>` o `[INDEX] insertado n/N`, por cada lote):

```text
2026-09-01 12:00:00 [PREFLIGHT] modelo sentence-transformers/all-MiniLM-L6-v2 disponible en ollama
2026-09-01 12:00:00 [LOAD] 234 documentos cargados
2026-09-01 12:00:01 [CLEAN] 234 documentos normalizados
2026-09-01 12:00:02 [TAG] 16 fuentes etiquetadas
2026-09-01 12:00:03 [CHUNK] 16214 chunks | min 40 · p25 612 · media 745 · p75 998 · max 1000 · cortos(<50) 12
2026-09-01 12:00:05 [EMBED] 16214 vectores de 384 dims (EMBED_DIM declarado: 384)
2026-09-01 12:00:06 [TSD] scoring + dedup
[DEDUP] umbral 0.93 -> descarta 1437 de 16214 (8.9%) | passthrough (política exacta) 14040 descartados exactos 0 · semánticos 2174
[INDEX] insertado 14777/14777 (14777 vectores, 10.2 s) (Restante: 0m 0s)
[INDEX] colección 'deporte_municipal': 14777 vectores (14777 en esta inserción)
2026-09-01 12:00:07 [FIN] pipeline completado en 261.05 s
2026-09-01 12:00:07 [INFORME] informe generado: output\\informe_index_0a1b2c3d_20260901_1200.md
```

Además, la función `ejecutar_pipeline` devuelve un diccionario de métricas (claves `num_documentos_cargados`, `num_chunks_pre_dedup`/`num_chunks_post_dedup`, `chunks_descartados`, `dim_embedding`, `dim_declarado`/`dim_modelo`, `chunk_stats`, `fuentes`, `integrity`, `scoring` —incluye cobertura por categoría y los tres tags principales—, `dedup`, `indice`, `preflight`, `fases` y `tiempo_total_s`) que alimenta directamente el informe y la evaluación.

### Parámetros opcionales del pipeline

La línea de comandos se invoca así:

```text
python -m src.pipeline [-h] [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP]
    [--persist-dir PERSIST_DIR] [--collection COLLECTION] [--recreate-index] rutas [rutas ...]
```

Los parámetros disponibles son los siguientes:

- `rutas`: archivos o carpetas a indexar (por ejemplo, `data`); no tiene valor por defecto y es obligatorio indicarlo.
- `--chunk-size`: longitud del chunk; por defecto toma el valor de `CHUNK_SIZE` definido en el archivo `.env`.
- `--chunk-overlap`: solapamiento entre chunks; por defecto toma el valor de `CHUNK_OVERLAP` definido en el archivo `.env`.
- `--persist-dir`: directorio donde reside ChromaDB; por defecto toma el valor de `CHROMA_DIR` definido en el archivo `.env`.
- `--collection`: nombre de la colección; por defecto toma el valor de `COLLECTION_NAME` definido en el archivo `.env`.
- `--recreate-index`: borra la colección antes de indexar; por defecto, no se borra.

### Informe de indexación (archivo `output/informe_index_<GUID8>_YYYYMMDD_HHMM.md`)

Cada ejecución escribe este informe (salvo cuando se desactiva con `informe=False` en las pruebas): una lectura única que responde a **qué parámetros determinaron esta ejecución** y **cómo quedaron las métricas**, en siete secciones (parámetros, preverificación, métricas por fase, troceado, índice final, bloque TSD y señales con criterios de decisión). Cada fila de la tabla de parámetros corresponde a un parámetro que realmente influyó; las credenciales no se vuelcan nunca.

Para la descripción ampliada —las reglas de la tabla de parámetros, la interpretación sección a sección y la relación de señales (condición y acción recomendada)—, consultar [`docs/ANEXO_OUTPUTS.md`](docs/ANEXO_OUTPUTS.md).

### Validación (scripts y pruebas)

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

## Configuración (archivo `.env`) sin tocar los archivos de código

Todo lo configurable reside en `config.py`, que lee el archivo `.env` con valores por defecto (los de la tabla del ejemplo son los de `.env.example`). Para empezar basta con editar estas variables en el archivo `.env`:

- `EMBED_PROVIDER` (valor por defecto, `ollama`): proveedor de embeddings, entre `ollama`, `huggingface` y `google`. Las variables `TAG_PROVIDER` y `GEN_PROVIDER` funcionan de la misma forma; el etiquetado hereda el valor de generación cuando está vacío.
- Las variables de modelo por proveedor e interruptor, esto es, `<PROVEEDOR>_EMBED_MODEL`, `<PROVEEDOR>_TAG_MODEL` y `<PROVEEDOR>_GEN_MODEL` (con sus valores por defecto en `.env.example`): modelo de cada interruptor por proveedor (Ollama, HuggingFace o Google).
- `OLLAMA_BASE_URL` (valor por defecto, `http://127.0.0.1:11434`): solo en caso de usar el proveedor Ollama.
- `HF_DEVICE` (valor por defecto, `cpu`): solo en caso de usar HuggingFace; admite los valores `cpu` y `cuda` (este último exige una tarjeta NVIDIA). La variable `HF_TOKEN` solo es necesaria para los modelos restringidos.
- `GOOGLE_API_KEY` (valor por defecto, vacía): solo en caso de usar Google.
- `CHUNK_SIZE` y `CHUNK_OVERLAP` (valores por defecto, `1000` y `150`): parámetros del troceado.
- `EMBED_DIM` (valor por defecto, `384`): tope de la dimensión del índice; la dimensión final es el mínimo entre la dimensión del modelo y esta constante.
- `TAG_SCORING_DEDUP` (valor por defecto, `true`): activa o desactiva el bloque TSD completo.
- `DEDUP_UMBRAL` (valor por defecto, `0.93`): umbral de similitud coseno para la deduplicación semántica.

Para el resto de variables (modelos por proveedor e interruptor, constante `EMBED_BATCH_SIZE`, memoria caché offline de la preverificación con las constantes `EMBED_DIM_MAX_*`, constante `EXPORT_EMBEDDINGS`, y las constantes `CHROMA_DIR` y `COLLECTION_NAME`), la referencia completa está en [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md) para los valores que resuelve `config.py` y en [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md) para el efecto que tiene cada constante; la plantilla canónica, con cada valor por defecto comentado, es el archivo [`.env.example`](.env.example).

**Importante:** si se cambian las constantes `EMBED_*` o `CHUNK_*`, o el propio corpus, hay que ejecutar de nuevo el pipeline con el flag `--recreate-index`. Los vectores son función directa del texto troceado; un índice no regenerado produce resultados inválidos.

## Cómo funciona cada fase

- **Preverificación** (módulo `src/embed.py`): comprueba que el modelo `EMBED_MODEL` existe en el proveedor indicado; se interrumpe si no existe, y degrada a la memoria caché `EMBED_DIM_MAX_*` en el caso de que no pueda verificarse en línea.
- **Carga** (módulo `src/load.py`): carga archivos PDF, TXT, MD o CSV (archivo único o carpeta), con detección automática de la codificación, y siempre puebla los metadatos `source` y `file_hash` (resumen SHA-256 del archivo).
- **Capa CSV** (módulos `csv_advisor.py` y `csv_transform.py`): clasifica cada archivo CSV y lo convierte en documentos dotados de su política de deduplicación; la deduplicación de los archivos CSV nunca es semántica.
- **Limpieza** (módulo `src/clean.py`): normaliza saltos de línea, espacios, carácter de orden al inicio (BOM) y caracteres de control; no toca los metadatos.
- **Etiquetado** (módulo `src/tsd/tag.py`): el modelo de lenguaje etiqueta **una sola vez por fuente** mediante una taxonomía cerrada y propaga las categorías `doc_category`, los tags y la relevancia, además de los valores booleanos de cada tag (contrato de filtrado de la fase online).
- **Troceado** (módulo `src/chunk.py`): usa el divisor recursivo de texto por caracteres, dividiendo primero en párrafos y luego en frases y letras a falta de otro límite; hereda los metadatos del documento padre y respeta la pista `chunking_hint` de la capa CSV (entidades `no_chunk` y grupos ligeros: indexados íntegros, sin trocear).
- **Vectorización** (módulo `src/embed.py`): genera los vectores por proveedor, normalizados y con la dimensión garantizada (ver la sección de [características](#características-principales)).
- **Puntuación** (módulo `src/tsd/scoring.py`): asigna a cada chunk su puntuación semántica mediante una búsqueda de vecinos más cercanos con FAISS (dos vecinos).
- **Deduplicación** (módulo `src/tsd/dedup.py`): descarta chunks según la puntuación (método greedy); los archivos con claves exactas (los CSV) solo se deduplican por clave.
- **Indexación** (módulo `src/index.py`): inserta en ChromaDB por lotes, sanea los metadatos y verifica que los identificadores estén vivos en la colección.
- **Informe** (módulo `src/informe.py`): escribe el archivo `output/informe_index_<GUID8>_YYYYMMDD_HHMM.md` con los parámetros, las métricas y las señales.

Cada fase se apoya en los **metadatos** que dejó la fase anterior: el etiquetado los escribe, el troceado los propaga a los fragmentos, la puntuación los lee y los enriquece con la puntuación semántica, la deduplicación los consulta para ordenar el descarte y la indexación los sanea antes de persistirlos en ChromaDB. Para el detalle por funciones (firmas y comportamiento), consultar [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md); para el estado completo del pipeline (flujo de invocación y métricas por consola), la referencia [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md).

### Criterios de puntuación semántica (`semantic_score`)

Cada chunk recibe una puntuación semántica (un valor entre cero y uno) que combina cuatro señales aditivas, guardada en el metadato `metadata['semantic_score']`:

```text
puntuación semántica = 0.45 x relevancia del LLM + 0.20 x centralidad
                    + 0.20 x (1 − redundancia) + 0.15 x autoridad
```

- **La relevancia asignada por el LLM** (peso 0.45): es la valoración de relevancia, de cero a uno, que el LLM asignó a la fuente en la fase de etiquetado (¿responde a preguntas de tarifas, reservas, horarios y normas?). Pesada más que el resto porque es la única señal alineada directamente con el objetivo de RAG, y no solo con la geometría de los embeddings.
- **La centralidad** (peso 0.20): es el coseno del chunk contra el centroide de la colección (la media de todos los vectores), que recompensa el contenido representativo del corpus y penaliza lo marginal.
- **La no redundancia** (peso 0.20): mide cuánto se diferencia el chunk de su vecino más parecido (búsqueda con FAISS de dos vecinos, excluyendo el propio); si el contenido ya existe en otro chunk, no aporta señal nueva.
- **La autoridad** (peso 0.15): es un ajuste fino según el tipo de fuente (reglamento o normativa, valor 1.0; precios o tarifas, valor 0.9; agenda, valor 0.6; el resto, valor 0.7). Una norma pesa más que un archivo CSV genérico, pero no puede compensar una baja relevancia.

La redundancia se calcula con FAISS (búsqueda de dos vecinos) en vez de materializar la matriz completa de similitud, de manera que la memoria crece linealmente con el número de chunks y su dimensión, y no de forma cuadrática (aproximadamente 1,3 GB frente a 68 GB en la escala real). La puntuación alimenta la deduplicación greedy de la fase DEDUP y, en el mapa de ruta, la reordenación de la recuperación. Para conocer el porqué del peso de cada parte y cómo se mide, consultar la sección de *scoring semántico* en [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md); para la implementación, [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md).

### Clasificación de los archivos CSV (`csv_advisor`)

Los archivos CSV no se tratan igual que el resto: antes de trocear, el módulo `src/csv_advisor.py` decide el tratamiento de cada uno (el registro `[CSV]` se muestra por consola). Son dos niveles de decisión, de lo más económico a lo más costoso: en primer lugar, la **receta de la fuente conocida** (la constante `RECETAS_CONOCIDAS`, con una confianza de 0.99); y, si el esquema cambió o la fuente es nueva, la **heurística por perfil estructural** (cardinalidades, densidad numérica y columnas de identificador, medida, temporal o narrativa). Un archivo CSV desconocido queda en la política conservadora `unknown_csv` o `exact_only` (confianza 0.4): no rompe nada y queda registrado para su revisión. La deduplicación de los archivos CSV **nunca es semántica**; se realiza por clave exacta. La tabla de decisiones del corpus actual, la distinción tablas de entidad frente a tablas de hechos y el glosario de los registros están en [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md).

## Corpus y fuentes

Corpus de la sede de **datos abiertos del Ayuntamiento de Madrid** (descargado en septiembre de 2026). Datos públicos y de uso educativo:

- El archivo `PreciosPublicos2026.pdf` procede de [Precios públicos centros deportivos 2026](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/ficheros/TarifasD/PreciosPublicos2026.pdf).
- El archivo `Tarifas_deportivas.pdf` procede de [Tarifas de servicios en centros deportivos](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Colecciones/ficheros/TarifasD/Tarifas_deportivas.pdf).
- El archivo `reglamento_instalaciones.pdf` es el [Reglamento de instalaciones deportivas](https://sede.madrid.es/eli/es-md-01860896/reg/2012/10/15/(1)/dof/spa/pdf).
- El archivo `PiscinasAireLibre2026.pdf` son las [Piscinas de verano 2026 aire libre](https://www.madrid.es/UnidadesDescentralizadas/Deportes/EspecialInformativo/Verano2026/ficheros/PiscinasAireLibre2026.pdf).
- El archivo `DecretoAnulacionReservasConCoste.pdf` es el [Decreto de anulación de reservas con coste](https://www.madrid.es/UnidadesDescentralizadas/Deportes/ContenidoGenerico/ContenidoGenerico2024/Ficheros/DecretoAnulacionReservasConCoste.pdf).
- Los archivos `normativaGeneral46jdm.pdf` y `BasesDeportesEquipo47jdm_1Sp.pdf` son la [Normativa de los Juegos Deportivos Municipales](https://www.madrid.es/UnidadesDescentralizadas/Deportes/EspecialInformativo/46%20Juegos%20Deportivos%20Municipales%252025-26/normativas/normativaGeneral46jdm.pdf).
- El archivo `20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf` es la [infografía del Abono Deporte Madrid](https://www.madrid.es/UnidadesDescentralizadas/Deportes/Faq/ficheros/infografias%20faq%202025/20250912_Infograf%C3%ADaC%C3%B3moAdquirirOrenovarUnADM.pdf).
- El archivo `200186-0-polideportivos.csv` corresponde al conjunto de datos [Polideportivos](https://datos.madrid.es/dataset/200186-0-polideportivos).
- El archivo `200215-0-instalaciones-deportivas.csv` corresponde al conjunto de datos [Instalaciones deportivas básicas](https://datos.madrid.es/dataset/200215-0-instalaciones-deportivas).
- El archivo `210227-0-piscinas-publicas.csv` corresponde al conjunto de datos [Piscinas públicas](https://datos.madrid.es/dataset/210227-0-piscinas-publicas).
- El archivo `300390-0-areas-deportivas.csv` corresponde al conjunto de datos [Áreas de actividades deportivas](https://datos.madrid.es/dataset/300390-0-areas-deportivas).
- El archivo `212504-0-agenda-actividades-deportes.csv` corresponde al conjunto de datos [Agenda de actividades deportivas](https://datos.madrid.es/dataset/212504-0-agenda-actividades-deportes).
- El archivo `300085-0-deportes_abonos.csv` corresponde al conjunto de datos [Abonados en centros deportivos](https://datos.madrid.es/dataset/300085-0-deportes_abonos).
- El archivo `300097-0-deportes-descuentos.csv` corresponde al conjunto de datos [Descuentos en instalaciones](https://datos.madrid.es/dataset/300097-0-deportes-descuentos).
- El archivo `211549-0-juegos-deportivos-actual.txt` corresponde al conjunto de datos [Juegos deportivos municipales vigente](https://datos.madrid.es/dataset/211549-0-juegos-deportivos-actual).

Los archivos CSV pueden regenerarse descargando cualquiera de los conjuntos de datos del [grupo deporte de datos.madrid.es](https://datos.madrid.es/group/deporte); el cargador los recarga igual aunque cambien de esquema, siempre que se identifiquen adecuadamente.

## Documentación técnica

- [`docs/ANEXO_FUNC.md`](docs/ANEXO_FUNC.md): **funciones de cada archivo** (módulos fuente, bloque TSD, capa de CSV, scripts y pruebas), con firmas y comportamiento clave.
- [`docs/FT_CORPUS_DATA.md`](docs/FT_CORPUS_DATA.md): estado completo del pipeline: módulos, flujo de invocación, proveedores y métricas por consola.
- [`docs/ANEXO_CHUNKING.md`](docs/ANEXO_CHUNKING.md): cómo cambiar el nivel de troceado y la validación obligatoria (incluye errores comunes).
- [`docs/ANEXO_OUTPUTS.md`](docs/ANEXO_OUTPUTS.md): salidas del pipeline, esto es, el informe de indexación (secciones, reglas de parámetros y señales con acción recomendada) y el archivo `output/embeddings.json`.
- [`docs/HIGHLIGHTS_CORPUS_DATA.md`](docs/HIGHLIGHTS_CORPUS_DATA.md): los apartados superados frente al enunciado y la guía de partida (preverificación, bloque TSD, dimensión garantizada e informe).

## Roadmap

- [x] Corpus multi formato (PDF, CSV y TXT).
- [x] Pipeline offline completo con bloque TSD.
- [x] Proveedor múltiple (Ollama, HuggingFace y Google).
- [x] Índice en ChromaDB verificado y regenerable.
- [ ] Fase online (recuperación): el flag `--query` con filtrado por `doc_category` y por los booleanos de cada tag (ya presentes en el índice mediante el filtrado `where` de Chroma) y reordenación por la puntuación semántica.
- [ ] Generación con RAG y abstención: el flag `--ask`.
- [ ] Interfaz de Streamlit: chat y chunks visibles, y tabla de métricas.

## Solución de problemas

- **El pipeline se interrumpe con el mensaje `PREFLIGHT: el modelo 'X' NO está disponible en <proveedor>`**. Causa probable: el modelo no existe en el proveedor seleccionado (error tipográfico, o Ollama no cuenta con él descargado). Solución: arrancar el servicio con el modelo descargado (por ejemplo, mediante el comando `ollama pull <modelo>`) o corregir la constante `EMBED_*_MODEL` en el archivo `.env`.
- **El pipeline emite el aviso `PREFLIGHT: no se pudo verificar online ...`**. Causa probable: red cortada o ausencia de clave API; no fue posible verificar el modelo (se asume disponible a través de la memoria caché `EMBED_DIM_MAX_*`). Es un mensaje informativo y el proceso continúa; para blindarlo completamente, conviene declarar la constante `EMBED_DIM_MAX_<proveedor>` en el archivo `.env`, arrancar el servicio y configurar la clave `GOOGLE_API_KEY`.
- **El pipeline emite el aviso `AVISO: EMBED_DIM (X) es mayor que la dim generada (Y)`**. Causa probable: el valor de `EMBED_DIM` es mayor que la dimensión que genera el modelo. El índice se creó con la dim generada (`Y`, las dimensiones que genera el modelo) y el valor declarado queda mostrado por separado en la tabla de parámetros (`EMBED_DIM_DECLARADO`). Solución: fijar `EMBED_DIM` en la dimensión del modelo (por ejemplo, `384` para el modelo `all-MiniLM-L6-v2`) en el archivo `.env`.
- **El pipeline emite el aviso `INFO: EMBED_DIM (X) es menor que la dim máxima del modelo (Y)`**. Causa probable: el valor de `EMBED_DIM` es menor que la dimensión máxima declarada. Es un mensaje informativo: el índice se creó con `X` dimensiones y el modelo puede manejar hasta `Y`; para indexar con más dimensiones, conviene aumentar `EMBED_DIM` y regenerar.
- **Ollama responde sin el campo de embeddings**. Causa probable: versión antigua del servicio. Solución: actualizar Ollama a al menos la versión 0.9 (el punto de servicio por lotes `/api/embed` lo exige) o cambiar a `EMBED_PROVIDER=huggingface`.
- **ChromaDB emite el error de tamaño de lote máximo superado** (mensaje `greater than max batch size`). Causa probable: colección dañada (el código ya inserta por lotes). Solución: borrar el directorio `output/` y volver a generar el índice.
- **La fase de etiquetado usa la categoría por defecto `instalaciones`**. Causa probable: el LLM devolvió un objeto JSON malformado. Solución: revisar que el modelo de `TAG_PROVIDER` sea capaz de emitir JSON válido. Para el pipeline no es un error: se marca en la consola y el proceso continúa.
- **La deduplicación descarta casi todo el corpus** (por ejemplo, más del noventa por ciento con `DEDUP_UMBRAL` fijado en 0.93). Causa probable: el umbral es ajustable y, en los archivos CSV con chunking por fuente, las fuentes repetitivas se deduplican a gran escala. Solución: bajar el umbral (por ejemplo, `0.85`) para conservar más chunks, o subirlo (por ejemplo, `0.95`) para deduplicar más.
- **Se cambió el troceado o el modelo y la recuperación falla**. Causa probable: el índice no fue regenerado. Solución: ejecutar `python -m src.pipeline data --recreate-index`.
