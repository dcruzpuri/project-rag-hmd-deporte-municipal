# Highlights del bloque `feature/corpus-data`

**Objetivo:** documentar qué aporta este bloque de trabajo para la fase de corpus e indexación offline (la primera y la segunda partes de la rama `feature/corpus-datos`).

La línea base que fija el enunciado es: cargar un corpus (de dos o más formatos), limpiarlo, trocearlo, vectorizarlo (embeddings), indexarlo en ChromaDB con el metadato de origen (la clave `source`), mantener la indexación separada de la consulta, registrar por consola (logging básico) y poder regenerar el índice. Lo que sigue a continuación marca, punto por punto, **lo mínimo exigido** y **lo que el bloque supera**.

## 1. El corpus (primera parte)

- **El número de formatos distintos**: el enunciado exige dos como mínimo. Se aporta **tres** en el directorio `data` (ocho PDF, siete CSV y un TXT); además, el cargador soporta también MD (markdown).
- **La mezcla de texto y de datos estructurados**: el enunciado la señala como preferible. Se aporta: los PDF normativos y de tarifas (texto) combinados con los CSV de datos abiertos estructurados y la agenda en TXT.
- **La documentación de las fuentes**: el enunciado pide enlaces y fecha. Se aporta un corpus real descargado de la sede de datos abiertos (datos.madrid.es), que cubre polideportivos, instalaciones, piscinas, abonos, descuentos, tarifas, reglamento y decretos de reserva.

**Más allá del mínimo:**

- Los archivos CSV no se alimentan ya del simple supuesto "una fila equivale a un documento plano": antes de trocear existe una capa de **clasificación diferencial** (los módulos `src/csv_advisor.py` y `src/csv_transform.py`) que decide el tratamiento de cada fuente (documento de entidad, cuando cada fila es una entidad de identificador estable y se deduplica por clave exacta; documento agrupado, cuando se trata de una tabla de hechos repetitiva y se deduplica por grupo; o documento por fila como caso refugio). Los metadatos (la política de deduplicación, la clave de entidad, la clave de grupo y la pista de troceado) viaja con el chunk hasta el índice, y **la deduplicación de los CSV nunca es semántica** (de modo que no se pierde información de entidades casi idénticas) sino por clave exacta.
- La carga soporta **un archivo o una carpeta recursiva** mediante un registro de extensiones (la constante `_LOADERS`): añadir un formato nuevo equivale a añadir una entrada a ese diccionario, sin tocar el resto.
- El metadato `source` está **siempre poblado**, aunque el cargador no lo haya fijado (la definición por omisión está en `load.py`).
- El metadato `file_hash` (el resumen SHA-256 del archivo) identifica el origen de forma unívoca: es estable entre indexaciones y detecta contenido nuevo cuando se descarga repetidamente un archivo del mismo nombre.

## 2. La preverificación: validación preventiva antes de empezar

Una **fase de preverificación** es un mecanismo de validación preventiva diseñado para bloquear errores, inseguridades o configuraciones incorrectas **antes de que afecten a la ejecución principal**. El término se usa en Python al menos en tres contextos, todos con la misma idea de puerta de control previa:

1. **Seguridad y carga de complementos**: mecanismos de este tipo validan el manifiesto de un complemento (seguridad, permisos y origen) antes de importar su código; es una puerta basada en archivos inactivos (JSON) previa a la importación real, para que nada malicioso ni no autorizado entre en el proceso. Dentro de este documento, se trata de una referencia conceptual; no es una dependencia que este proyecto utilice.
2. **Despliegue y calidad del código**: algunas herramientas de línea de comandos examinan el conjunto de código antes de enviarlo a producción (por ejemplo, variables de entorno erróneas, claves de API filtradas, impresiones de depuración o controles de seguridad ausentes): un estado «listo para volar» antes del despliegue.
3. **Inicialización del entorno**: la pre-inicialización del intérprete (la estructura de configuración y su carga): la memoria, las codificaciones y la configuración fijadas antes de que el núcleo se cargue por completo. Se observa en el arranque del intérprete de Python (la estructura de pre-configuración y su función de inicialización, poblada desde las opciones de arranque y visible en tiempo de ejecución a través de las banderas del sistema). Como los anteriores, es una referencia conceptual, no una dependencia de este proyecto.

Este proyecto **aplica ese mismo espíritu a la indexación**: el pipeline empieza ahora por la **fase de preverificación, situada antes de la carga**, que garantiza que el modelo de embeddings está disponible en el proveedor elegido (mediante la ruta `/api/tags` en Ollama, mediante la comprobación de existencia del repositorio en HuggingFace o mediante el listado de modelos en Google) y **se interrumpe** (lanza un error de tiempo de ejecución) cuando no existe. Si la comprobación en línea no es posible (red cortada, Ollama apagado o ausencia de clave API), se degrada con un aviso y recurre a la memoria caché offline declarada en el archivo `.env` (las constantes `EMBED_DIM_MAX_*`, y el indicador de verificación toma el valor de memoria caché), en vez de romper el pipeline. El resultado (si se verificó en línea o mediante caché, la dimensión máxima declarada y el aviso) se refleja en la consola y en la segunda sección del informe.

## 3. El pipeline offline (segunda parte) — más allá de la cadena básica

Se ha ido más allá de lo que es la cadena básica (carga, limpieza, troceado, vectorización e indexación) y se inserta un **bloque etiquetado-puntuado-deduplicado** (TSD), porque mejora la calidad de la recuperación antes de que los chunks entren al índice, además de generar un informe tras la indexación que resume la parametrización y refleja métricas para la decisión. Es decir, las fases se encadenan de forma que, por delante de la cadena básica, se añade **la preverificación**; entre la vectorización y la indexación, **etiquetado, puntuación y deduplicación**; y al final, **el informe** (cada una de ellas es una fase adicional sobre la cadena básica).

- **El etiquetado** (`tsd/tag.py`): etiquetado semántico con un modelo de lenguaje usando una taxonomía cerrada (la categoría del documento, los tags y la relevancia). Etiqueta **una vez por fuente** (una única llamada al modelo por documento, no por página) y propaga el resultado a todos sus chunks, lo que lo hace luego barato y consistente.
- **La puntuación** (`tsd/scoring.py`): la puntuación semántica (`semantic_score`) por chunk, que combina la relevancia asignada por el modelo, la **centralidad** (el coseno contra el centroide), la **redundancia** (el coseno máximo con el resto, mediante FAISS: el consumo de memoria es proporcional al número de filas por su dimensión, en vez de proporcional al cuadrado del número de filas) y la **autoridad** de la fuente; vuelca sus métricas en la referencia de información.
- **La deduplicación** (`tsd/dedup.py`): la deduplicación **consciente de política**; los chunks con política exacta (por clave de entidad, por grupo o estrictamente por clave) se deduplican por clave y **nunca por coseno**, y los demás pasan por la deduplicación por '_recorte-greedy_' (recorte voraz) ordenada por la puntuación semántica con umbral ( `DEDUP_UMBRAL` ) configurable (**FAISS incremental**).
- **El informe**: al terminar, la función de orquestación del pipeline de ingesta ensambla el diccionario de métricas y genera el informe [sección 4](#4-el-informe-de-indexación-en-markdown-srcinformepy); se puede saltar en las pruebas.

**Más allá del mínimo:** todo el bloque es opcional mediante un conmutador (la constante `TAG_SCORING_DEDUP`), y el pipeline devuelve un diccionario mucho más rico que el mínimo (los tiempos por fase, los resúmenes de cada fase, las métricas de puntuación y deduplicación, la dimensión y la preverificación).

## 4. El informe de indexación en markdown (`src/informe.py`)

Como extra tras la ingesta, el informe de indexación se genera **automáticamente al terminar** con todo lo necesario para decidir qué parámetros conviene tocar:

- **Los parámetros aplicados** (los proveedores, los modelos, la dimensión efectiva, el troceado, el bloque TSD, el umbral, la colección y el regenerado): qué ejecución fue la que dejó ese índice.
- **La preverificación**: cómo se verificó el modelo (en línea o mediante memoria caché), la dimensión máxima declarada y el aviso.
- **Las métricas por fase**, desde la preverificación hasta la terminación, con el tiempo y el resumen de cada una.
- **El troceado** (el mínimo, el percentil veinticinco, la media, el percentil setenta y cinco y el máximo; los chunks de menos de cincuenta caracteres constituyen ruido probable a investigar) y **el índice final** (los vectores insertados frente a los totales de la colección, la dimensión y si se regeneró).
- **El bloque TSD**: la puntuación (el mínimo, la media y el máximo; el porcentaje de chunks con puntuación mayor o igual a 0,6; y las medias de centralidad y redundancia) y la deduplicación (antes y después, el descarte exacto frente al semántico y el porcentaje total).
- **Las señales y los criterios de decisión**: heurísticas automáticas sobre las métricas (la deduplicación que descarta más del cincuenta por ciento, la deduplicación que no descarta nada, la dimensión desajustada, la preverificación por memoria caché, menos del cincuenta por ciento de chunks buenos, entre otras), cada una con la acción concreta que recomienda (bajar o subir el umbral, tocar la dimensión, revisar los cargadores, entre otras).

El módulo **solo renderiza** datos (no lee la configuración ni produce efectos secundarios), de modo que su salida se prueba de forma aislada, y la prueba de extremo a extremo corre con un proveedor simulado (sin red ni Ollama). Concretamente:

- **«Se prueba de forma aislada»**: el informe es una función pura (su entrada es el diccionario de métricas que devuelve la orquestación y su salida es la cadena en markdown), de modo que las pruebas fabrican ese diccionario a mano y solo verifican el texto generado (las secciones, los valores y las señales presentes); no hay base de datos, archivo de configuración ni red que preparar.
- **«El extremo a extremo con proveedor simulado»**: la prueba de extremo a extremo del pipeline usa un proveedor de embeddings simulado (devuelve vectores deterministas de dimensión fija, sin llamar a Ollama ni a ninguna API) y escribe el índice en un directorio temporal; así el flujo completo, de la preverificación hasta el informe, se valida sin red ni Ollama instalado.

## 5. La dimensión garantizada

Aquí cabe suponer _erróneamente_ que la dimensión del modelo y la constante `EMBED_DIM` coinciden, o a lo sumo la constante es menor a la dimensión que soporta el modelo. En este proyecto, la dimensión del índice es **siempre** el mínimo entre la dimensión que maneja el modelo y la constante `EMBED_DIM`, con una doble protección:

- **Una pista al proveedor en el servidor**: Ollama (el campo `dimensions` en la petición) y HuggingFace (el parámetro `truncate_dim`, mediante recorte tipo Matryoshka); Google la acepta y la ignora de forma explícita.
- **La garantía real en el cliente** (la función `_corte_dim` en `src/embed.py`): la función de vectorización es la **única puerta** por la que pasan el índice y la consulta futura, y recorta el prefijo y lo renormaliza si el proveedor devuelve más dimensiones de las declaradas. La métrica coseno se mantiene.

> El **recorte tipo Matryoshka** es una técnica de aprendizaje de representación anidada: el modelo se entrena para que cualquier prefijo del vector siga siendo un vector válido; por eso, recortar a las primeras dimensiones declaradas y renormalizar no degrada la similitud coseno. Se revisa en la función `_corte_dim` de `src/embed.py` (recorta el prefijo y renormaliza el vector) y en la función de vectorización, que la aplica como única puerta del índice y de la consulta.

El chequeo de dimensiones se convierte en mensajes por niveles: cuando la dimensión declarada es menor que la dimensión máxima del modelo, se emite un mensaje **informativo** (se indexa con la dimensión declarada); cuando es mayor que la dimensión generada, se emite un **aviso** y el índice se crea con la dimensión generada (la que maneja el modelo), sin mutar la constante `EMBED_DIM`: las tres dimensiones (declarada/modelo/efectiva) se muestran por separado en la tabla de parámetros y en el informe. Con ello se evita que «el índice se creara con una dimensión que el modelo no soporta»: **quedó solucionado de raíz**.

## 6. El índice en ChromaDB (`src/index.py`)

- **Chroma persistente con el metadato de origen**: se aporta, **y además saneado** (Chroma no admite valores vacíos ni listas, por lo que se normalizan).
- **El regenerado del índice**: se aporta mediante la opción de recreación, con borrado seguro de la colección.
- **La verificación simple**: se aporta mediante la verificación de la inserción (los identificadores insertados deben existir en la colección).

**Como extra:** la función de indexación devuelve ahora una tupla de dos elementos (los identificadores vivos y el total de la colección), de manera que el informe muestre ambas cifras y se detecte un índice parcialmente regenerado. El cliente y la creación de la colección son funciones reutilables (la obtención del cliente de Chroma y la creación de la colección), pensadas para que la fase de recuperación (la consulta vectorial, el filtro por la categoría del documento y la reordenación por la puntuación semántica) sea posible sin tocar el indexado.

## 7. Los embeddings multi-proveedor (`src/embed.py`)

Para el apartado API de LLM y embeddings se plantean tres proveedores distintos:

- **Tres proveedores intercambiables por variable de entorno** (la constante `EMBED_PROVIDER`): Ollama (la ruta por lotes `/api/embed`), HuggingFace (el cargador de sentence-transformers, con carga perezosa y memoria caché) y Google (la ruta por lotes de la API de servicio web). Añadir un proveedor equivale a añadir una función y una línea al diccionario de proveedores.
- **La coherencia de la métrica**: los embeddings normalizados y la colección con métrica coseno hacen que la similitud sea coseno real en toda la cadena (vectorización, puntuación, deduplicación y recuperación).
- **El tope máximo de dimensión por proveedor** [sección quinta](#5-la-dimensión-garantizada) y **la preverificación de disponibilidad** [sección segunda](#2-la-preverificación-validación-preventiva-antes-de-empezar), con memoria caché offline por medio de las constantes `EMBED_DIM_MAX_OLLAMA`, `EMBED_DIM_MAX_HF` y `EMBED_DIM_MAX_GOOGLE`: la preverificación declara la dimensión máxima que maneja el modelo incluso sin red ni clave API.
- **La exportación a JSON** (la función de vectorización y la función de exportación): vuelca el texto, los metadatos y el vector a `output/embeddings.json` para inspección y depuración, lo que ayuda al experimento de troceado del informe.
- La función de consulta vectorial (la vectorización de una pregunta) ya está lista para la fase online (la condición RAG: el mismo modelo que al indexar) y con la misma garantía de dimensión/ que el índice.

## 8. Robustez y correcciones aplicadas sobre el código de esta rama

Son correcciones que hacen que el pipeline **funcione de punta a punta** (evitar el fallo en la fase de indexación, la pérdida de datos o el arranque contra un proveedor no soportado o inexistente):

- **Los identificadores de chunk globales y únicos** (pipeline.py): se evita que, al ser el índice de chunk secuencial por documento, un archivo CSV con la misma fuente genere identificadores duplicados que ChromaDB habría sobreescribido. Aquí el identificador es único (un índice global) y ninguna fila se pierde.
- **La preverificación con aborto a tiempo**: un modelo de embeddings mal escrito o no descargado no rompe la ejecución en la fase de vectorización tras minutos de carga y etiquetado; el pipeline se interrumpe en segundos con el mensaje correspondiente (el modelo no está disponible en el proveedor).
- **La palabra clave de metadatos correcta** en la inserción de la colección (index.py) y la firma de la función de indexación consistente con la llamada que la invoca (pipeline.py).
- **La corrección de la toma de la fuente** (tag.py): la lectura del metadato de origen estaba protegida de forma que un valor vacío no rompiera la propagación de las etiquetas a todas las páginas o filas.
- **Los guardas defensivos**: el centroide degenerado en la puntuación (con norma cero), la deduplicación sobre una lista vacía, un solapamiento de cero no se interpreta como valor falso, y un error explícito si el proveedor no devuelve el campo esperado. Son parte de un requisito imprescindible en tiempo de ejecución.
- **La deduplicación por política exacta para los CSV** (el bloque TSD y el asesor): la deduplicación semántica por coseno ya no descarta entidades casi idénticas (dos piscinas del mismo barrio no son duplicados semánticos que apenas difieren en unos pocos datos); la deduplicación semántica solo opera sobre los chunks no estructurados.
- **El registro por consola seguro** (evita los caracteres que rompen la terminal en Windows) y la verificación real de la inserción, en lugar de un recuento acumulado que podría no decir verdad al regenerar.

## 9. El troceado con criterio (nivel dos) y su validación

- El divisor recursivo de texto por caracteres con separadores **semánticos** (párrafo, frase, espacio y letra): el divisor «multiusos» recomendado como punto de partida.
- El módulo de troceado es el **puente entre el etiquetado y el troceado**: hereda la metadata del documento padre (la categoría del documento, los tags, la relevancia asignada por el modelo y, ahora, también la política de deduplicación, la clave de entidad y la clave de grupo de la capa CSV) a cada chunk, de modo que viaja sin coste adicional en tokens hasta el índice. Además **respeta la pista del asesor** (la de no trocear y la de troceado ligero): una fila de entidad de doscientos caracteres no pasa por el mismo proceso de corte que un reglamento de ochenta páginas, como regla lógica plausible.
- **La validación de coherencia** (`scripts/eval_coherencia_chunks.py`): mide la similitud coseno entre chunks adyacentes frente a la de pares aleatorios, para confirmar que el solapamiento mantiene la coherencia temática (la «Regla del troceado»: el dato ha de recuperarse con valor).
- Las pruebas offline del troceado (`tests/test_chunks.py`), que cubren el tamaño, el solapamiento, los metadatos y la preservación del contenido (los precios y los horarios no se rompen), además de las métricas para el experimento del tamaño y el solapamiento del troceado en el informe. 

```bash
python -m evaluar_rag_corpus.py [-h] --embeddings EMBEDDINGS --chunks CHUNKS [--eval EVAL] [--out OUT] [--dedup-threshold DEDUP_THRESHOLD] [--k K [K ...]] [--sample-pairs SAMPLE_PAIRS]
```

## 10. La configuración centralizada (`config.py`)

El conmutador de proveedores, los modelos, la dimensión, el tamaño de lote, el tamaño y el solapamiento del troceado, el umbral de deduplicación, el directorio y el nombre de la Colección **y la memoria caché offline de la preverificación** (las constantes `EMBED_DIM_MAX_OLLAMA`, `EMBED_DIM_MAX_HF` y `EMBED_DIM_MAX_GOOGLE`) viven en el archivo `.env` mediante el módulo `config.py`. El cambio de proveedor implica un cambio de variable, **sin tocar el código**. Existen tres conmutadores independientes por proveedor (embeddings, generación y etiquetado) con herencia por omisión: cuando el etiquetado no se define, hereda del proveedor de generación.

## Resumen en una frase

> La fase base (el corpus multi-formato y el pipeline de carga, limpieza, troceado, vectorización e indexación con Chroma) está **completo y funcional**; lo que se supera con esta rama de desarrollo es la **fase de preverificación** (la validación preventiva de la disponibilidad del modelo antes de arrancar, con memoria caché offline como fallback seguro), la **dimensión garantizada** (el mínimo entre la dimensión del modelo y la constante declarada), el **tratamiento diferencial de los archivos CSV** (la clasificación y la deduplicación por clave, nunca semántica), el **bloque etiquetado-puntuado-deduplicado** (la entrega de señal limpia al índice), **el informe de indexación con señales para decidir** y la abstracción multi-proveedor con la robustez suficiente para que el índice no pierda datos ni se rompa ante entradas anómalas.   
