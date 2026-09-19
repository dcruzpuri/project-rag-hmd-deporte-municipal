# Arquitectura RAG

## Gestión del flujo del pipeline

## Propósito

Documentar el flujo de datos, las responsabilidades de cada componente y las métricas internas utilizadas en el pipeline de ingesta de datos.

### I. Mapa de Responsabilidades de Componentes

| Módulo | Razón de Existencia | Función Principal | Descripción de Tarea |
| :--- | :--- | :--- | :--- |
| `src/pipeline.py` | Orchestrator Pipeline | `ejecutar_pipeline` | Controla la secuencia **lineal** de todo el pipeline, desde la ingesta hasta la generación del reporte final. |
| `src/load.py` | I/O Pipeline | `cargar_archivos` | Lee formatos crudos (PDF, TXT, MD, CSV). **Métrica de Trazabilidad:** Asigna `file_hash` (SHA-256 del contenido crudo) y `source` a cada `Document`. |
| `src/clean.py` | Pre-procesamiento | `limpiar` | Reduce el ruido (*no tokens/vectores*) en el contenido (`page_content`) en tres pasos: **unificación de saltos de línea, consolidación de espacios y eliminación de caracteres invisibles**. |
| `src/chunk.py` | División | `trocear` | Divide el contenido en `Documents`. Utiliza limpio  `RecursiveCharacterTextSplitter` con separadores jerárquicos, bypaseando el proceso si los metadatos lo exigen (p.ej. **entidades únicas**). |
| `src/tsd/tag.py` | Enriquecimiento / Metadatos | `etiquetar` | Atribuye metadatos semánticos (Taxonomía cerrada: `doc_category`, `tags`, `relevancia_llm`) a nivel de **Fuente** de datos. Llama a LLMs externos (con coste $$$ en provider). |
| `src/embed.py` | Vectorización (Puerta Única) | `embeddear` / `embeddear_consulta` | Convierte texto a embeddings. Es la capa crítica de seguridad: utiliza `EMBED_PROVIDER` (Ollama, HF, Google) y **garantiza** que la dimensión de salida nunca exceda `EMBED_DIM` mediante un recorte de prefijo y **normalización L2**. |
| `src/tsd/scoring.py` | Calificación | `puntuar` | Calcula `semantic_score` para cada chunk mediante una fórmula ponderada. Esta fórmula sintetiza 4 métricas: Relevancia LLM (Factor: 0.45)· Centralidad (Vector vs. Centroide Corpus) (Factor: 0.20) · Redundancia (1 - Similaridad Vecino Más Cercano) (Factor: 0.20) · Autoridad Fuente (Peso basado en `source`) (Factor: 0.15) |
| `src/tsd/dedup.py` | Limpieza Semántica | `deduplicar` | Elimina chunks potencialmente más. Respeta dos modos: 1. **Clave Exacta** (dedup por clave); 2. **Semántica:** Usa **FAISS*** y `DEDUP_UMBRAL` (Coseno) para asegurar la unicidad conceptual. |
| `src/retrieve.py` | Búsqueda | `recuperar` | Se ejecuta sobre ChromaDB. Toma la consulta, la vectoriza y ejecuta `query()` para obtener los top-k vecinos con su distancia. |
| `src/prompts.py` | Ingeniería de Prompt | `construir_prompt_desde_chunks` | Estructura el contexto recuperado (lista de chunks) en un bloque de texto legible y estructurado usando las `INSTRUCCIONES_SISTEMA`. |
| `src/logic.py` | Orquestador Online | `responder` | **Flujo RAG en tiempo real:** Orquesta las llamadas a `retrieve` $\rightarrow$ **Guardrail de Evidencia** (distancia > 0.65 $\Rightarrow$ abstención) $\rightarrow$ `generate` $\rightarrow$ Resultado final estructurado. |
| `src/informe.py` | Reporting | `generar_informe` | Toma el diccionario de estado del pipeline y lo renderiza en un Markdown detallado para auditoría y revisión de métricas. |

**(*)** **FAISS (Facebook AI Similarity Search)**. Es una librería de código abierto diseñada para la búsqueda eficiente de similitudes semánticas y vectores densos. Permite sistemas de recomendación y la búsqueda de imágenes 

### Flujo de Datos

El flujo es secuencial e interdependiente:

$$
\text{Carpetas/Archivos} \xrightarrow{LOAD} \text{Chunks Crudos} \xrightarrow{CLEAN} \text{Chunks Limpios} \xrightarrow{TAG} \text{Chunks Metadata Riqueza} \xrightarrow{CHUNK} \text{Chunks Finales} \xrightarrow{EMBED} \text{Embeddings Normalizados} \xrightarrow{SCORING} \text{Scores} \xrightarrow{DEDUP} \text{Chunks Post-Dedupl} \xrightarrow{INDEX} \text{ChromaDB Index}
$$

**Detalle del Flujo de Consulta (Runtime):**
1. **PREGUNTA** $\rightarrow$ `src/embed.py` (Vectorización).
2. $\rightarrow$ `src/retrieve.py` (Consulta Chroma $\rightarrow$ Top-K Chunks).
3. $\rightarrow$ **GUARDRAIL:** Chequeo de la distancia de simetría ($\text{distancia}\le 0.65$).
4. $\rightarrow$ `src/prompts.py` (Formateo de Contexto).
5. $\rightarrow$ `src/generate.py` (Llamada a LLM).
6. $\rightarrow$ Resultado Final.

### I. Funcionalidades y Métricas de Alto Impacto

#### A. Estrategias de Seguridad y Consistencia
1. **Dimensionalidad Consistente:** `embeddear()` es la única puerta de salida. Garantiza que *tanto* los embeddings del índice *como* los de la consulta sean dimensionalmente homogéneos, aplicando un recorte de prefijo y **normalización L2**.
2. **Trazabilidad del Origen:** Uso de `file_hash` (SHA-256) y `source` en **todos** los módulos para rastrear un chunk hasta su documento de origen.
3. **Guardrail de Evidencia:** Implementa un chequeo en `src/logic.py` que aborta la generación si la distancia mínima de los chunks es muy alta, forzando la abstención.

#### B. Métodologías Avanzadas de Procesamiento
| Métrica / Concepto | Módulo Responsable | Objetivo | Detalle Técnico |
| :--- | :--- | :--- | :--- |
| **Deduplicación Semántica** | `dedup.py` | Eliminar la redundancia conceptual. | Usa **FAISS** para buscar vecinos más cercanos (similitud coseno); si la similitud $\ge$ `DEDUP_UMBRAL`, el chunk se descarta. |
| **Puntuación Compuesta** | `scoring.py` | Evaluar la potencial utilidad de un chunk. | Fórmula ponderada de 4 factores: $\text{Score} = 0.45 \cdot \text{RelevanciaLLM} + 0.20 \cdot \text{Centralidad} + 0.20 \cdot (1 - \text{Redundancia}) + 0.15 \cdot \text{Autoridad}$. |
| **Metadata Estructurada** | `tsd/tag.py` | Codificar la taxonomía. | Usa un esquema estandarizado (`tag_<nombre>: True`) en los metadatos para permitir el filtrado preciso en la consulta vectorial. |
| **Autoridad (Weighting)** | `scoring.py` | Peso informativo estructural. | Asigna pesos en `AUTORIDAD` (documentos *reglamento* > *precios*). |

***

### II. Observaciones de Ingeniería Clave

*   **Separación de Roles:** La lógica está perfectamente desacoplada. El `pipeline.py` *solo* llama a los módulos; no contiene la lógica de negocio. Los módulos solo exponen funciones claras (clean, load, embed, etc.).
*   **Robustez:** El uso de *providers* (Ollama, HF, Google) en `embed.py` y `src/generate.py` permite la **simulación** de fallos de red o cuotas, implementando estrategias de `retry` (exponential backoff/retry) en el código, elevando la resiliencia.