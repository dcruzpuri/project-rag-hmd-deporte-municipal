# HT-CHUNK-01 — Guía técnica: cambio del nivel de chunking (How-to)

## 1. Objeto

El presente documento describe el procedimiento para modificar el nivel de chunking (o sus parámetros) del pipeline RAG del proyecto, así como los pasos de validación obligatorios tras la modificación. Se redacta como guía operativa de referencia para el equipo.

## 2. Alcance y referencias

*   **Aplicabilidad:** `src/chunk.py` (único punto del corte), `config.py`, `scripts/eval_coherencia_chunks.py` y `tests/test_chunks.py` .
    
*   **Referencias:** sección punto 6 de `HIGHLIGHTS_CORPUS_DATA.md` ("Chunking con criterio (Level 2) y validación") y escala de niveles `1 Character → 2 Recursive → 3 Document-specific → 4 Semantic → 5 Agentic` documentada en `FT_CORPUS_DATA.md` punto 6 .
    

## 3. Regla fundamental

**◬ ATENCIÓN**: Cualquier modificación del splitter o de `CHUNK_SIZE`/`CHUNK_OVERLAP` exige la **regeneración completa del índice**. Los vectores almacenados son función directa del texto troceado; un índice no regenerado tras el cambio produce resultados inválidos .

## 4. Estado inicial del sistema

*   **Nivel activo:** 2 (Recursive character), implementado en `src/chunk.py::trocear` mediante `RecursiveCharacterTextSplitter` con separadores `["\n\n", "\n", ". ", " ", ""]` .
    
*   **Validación:** `scripts/eval_coherencia_chunks.py` y `tests/test_chunks.py` .
    
*   **Indexación:** `ejecutar_pipeline(..., recreate_index=True)` .
    

### Niveles disponibles: costes y beneficios

| Nivel | Coste | Beneficio | Indicación |
| --- | --- | --- | --- |
| 1 · Character | Muy bajo (sin dependencias, trivial) | Rapidez extrema; útil como punto de contraste | Solo como experimento: es rígido y corta a ciegas  |
| 2 · Recursive | Bajo (nivel actual, sin dependencias extra) | Equilibrio entre velocidad y respeto a párrafos/frases | Nivel por defecto del proyecto  |
| 3 · Document-specific | Medio: MD no añade dependencias; las tablas de PDF requieren Unstructured y un pipeline de ingesta distinto | Preserva la estructura del documento (encabezados, tablas) | Corpus homogéneo y estructurado  |
| 4 · Semantic | Alto: batch extra de embeddings por documento fuente; es el nivel más caro | Cortes por cambio de tema; no rompe unidades de significado | Solo si el eval lo justifica  |
| 5 · Agentic | Muy alto: lento, caro y no determinista (índice difícil de reproducir) | Flexibilidad teórica de corte | No se recomienda  |

**Preparado en el venv (comprobado):** `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `MarkdownTextSplitter`, `MarkdownHeaderTextSplitter`, `SpacyTextSplitter`. **No están** `langchain_text_splitters.semantic` ni `SemanticTextSplitter`; el nivel 4 se implementa manualmente con `embeddear` .

## 5. Procedimiento

### 5.1. Preparación

1.  Se fijará el directorio de trabajo del corpus (`data/`) y el conjunto de preguntas de evaluación, que deberán permanecer invariantes durante el experimento .
    
2.  Se obtendrá un snapshot de referencia (nº de chunks, margen de coherencia, hit-rate) mediante:
    
    ```powershell
    python -m pytest tests/test_chunks.py::TestChunkMetrics -v -s
    python -m scripts.eval_coherencia_chunks
    ```
    
3.  Se documentará en el informe el valor inicial de `CHUNK_SIZE`/`CHUNK_OVERLAP` y el splitter empleado (se registra automáticamente en el `output/informe_index_(...).md` correspondiente).
    

### 5.2. Aplicación del cambio

Según el objetivo:

*   **Ajuste de parámetros (mismo nivel):** cambio en `.env` (global) o por parámetro de `trocear` (puntualmente al llamarlo por comando). La comprobación es `is not None`, por lo que `chunk_overlap=0` es un valor válido y respeta el 0 .
    
*   **Nivel 1 (contraste):** sustitución del bloque del splitter en `src/chunk.py` por `CharacterTextSplitter` .
    
*   **Nivel 3 (Document-specific):** selección del splitter por extensión manteniendo la firma de `trocear` (p. ej. `MarkdownTextSplitter` para `.md`). Las tablas de PDF requieren Unstructured (no instalado) y un pipeline de ingesta distinto; no se implementa hasta que el eval lo pida .
    
*   **Nivel 4 (Semantic):** implementación de breakpoints por similitud coseno entre frases adyacentes con `embeddear` y `numpy`; umbral sugerido 0.20–0.25 (más bajo = más breakpoints) .
    
*   **Nivel 5 (Agentic):** no se recomienda; únicamente como experimento opcional documentado .
    

### 5.3. Regeneración del índice

En Python ejecutar:

```python
ejecutar_pipeline("./data", recreate_index=True)
```

Equivalente directo por consola:

```powershell
python main.py ./data --recreate-index
```


### 5.4. Validación (obligatoria)

1.  **Coherencia de los cortes:** margen > 0.05 (el overlap mantiene coherencia temática) .
    
2.  **Distribución de longitudes:** documentada para el informe de chunking .
    
3.  **Retrieval** sobre las mismas preguntas con **al menos 2 valores de K**, anotando si el mejor hit mantiene sentido o si aparece ruido .
    
4.  **Regeneración del índice definitivo** .
    
5.  **Documentación** en `entregables/informe_decisiones.md`: nivel empleado, justificación, experimento `CHUNK_SIZE`/`CHUNK_OVERLAP` con métricas, y al menos 1 acierto + 1 abstención de la fase de generación .
    

## 6. Criterio de aceptación del cambio de nivel

Se adoptará un nivel superior únicamente si el eval evidencia una mejora medible (hit-rate o margen de coherencia) que justifique el coste adicional. En ausencia de mejora, se mantendrá el nivel previo y la decisión se documentará con las métricas correspondientes .

## 7. Errores comunes y su resolución

| Error | Consecuencia | Resolución |
| --- | --- | --- |
| Modificar el splitter o `CHUNK_SIZE`/`CHUNK_OVERLAP` sin regenerar el índice | Vectores obsoletos sobre un corte nuevo; resultados de retrieval inválidos | Regenerar con `ejecutar_pipeline(..., recreate_index=True)`  |
| Cambiar el corpus o las preguntas de eval durante el experimento | Resultados no comparables; experimento no reproducible | Fijar corpus y preguntas antes de empezar y guardar el snapshot de referencia  |
| Pasar `chunk_overlap=0` esperando que se aplique el valor por defecto | Confusión sobre el parámetro efectivo (el 0 se respeta) | La comprobación es `is not None`: para aplicar el default, omitir el parámetro  |
| Cambiar el modelo de embeddings sin regenerar el índice | Índice incompatible con los nuevos vectores | Regenerar el índice desde cero  |
| Sustituir el splitter global del nivel 3 sin mantener la firma de `trocear` | Se rompen los consumidores del módulo (pipeline, tests) | Seleccionar el splitter por extensión dentro de `src/chunk.py` conservando la firma  |
| Importar `SemanticTextSplitter` de `langchain_text_splitters` | `ImportError`: no está disponible en el venv | Implementar los breakpoints con `embeddear` + `numpy` (nivel 4 manual)  |
| Fijar el umbral del nivel 4 fuera de rango sin medir | Demasiados breakpoints (umbral bajo) o fusión de temas (umbral alto) | Medir coste y coherencia; documentar el umbral elegido en el informe  |
| Margen de coherencia < 0.05 tras el cambio | Overlap insuficiente; pérdida de coherencia temática | Aumentar `CHUNK_OVERLAP` o reducir `CHUNK_SIZE` y revalidar  |
| Aplicar el nivel 5 (Agentic) como camino principal | Índice lento, caro y no reproducible | Descartar; solo como experimento opcional documentado  |
| No documentar nivel, parámetros y métricas tras el cambio | Experimento no reproducible; memoria incompleta | Documentar en `entregables/informe_decisiones.md` según punto 5.4  |
| Modificar la lista de separadores de `RecursiveCharacterTextSplitter` (p. ej., usar la por defecto) | Cortes distintos a los del índice actual; vectores obsoletos | Regenerar el índice y revalidar coherencia  |
| Fijar `CHUNK_OVERLAP` >= `CHUNK_SIZE` | `ValueError` en el splitter o chunks degenerados | Mantener overlap < size y revalidar margen y distribución  |
| Cambiar `CHUNK_SIZE` sin revisar `CHUNK_OVERLAP` | El ratio de overlap cambia y puede romperse la coherencia | Recalcular el overlap y revalidar margen > 0.05  |
| Cambiar el valor de K entre ejecuciones del eval | Métricas de retrieval no comparables entre runs | Fijar al menos 2 valores de K en todas las ejecuciones y documentarlos  |
| Aplicar `MarkdownTextSplitter` a archivos no Markdown (p. ej., PDF) | Pérdida de estructura o cortes incorrectos | Seleccionar el splitter por extensión dentro de `trocear`; las tablas de PDF requieren Unstructured  |
| Modificar `scripts/eval_coherencia_chunks.py` durante el experimento | Métricas no comparables con el snapshot de referencia | No tocar el script de eval durante el experimento; documentar cualquier cambio posterior  |

## 8. Tabla de consulta rápida

| Operación | Modificación | Regeneración del índice |
| --- | --- | --- |
| Ajuste de parámetros (misma estrategia) | `CHUNK_SIZE`/`CHUNK_OVERLAP` en `.env` o parámetro de `trocear` | Obligatoria  |
| Contraste con nivel 1 | `CharacterTextSplitter` en `src/chunk.py` | Obligatoria  |
| Cambio de separadores (nivel 2) | Lista de separadores de `RecursiveCharacterTextSplitter` en `src/chunk.py` | Obligatoria  |
| Estructura Markdown (nivel 3) | Loader por extensión + `MarkdownTextSplitter` | Obligatoria  |
| Breakpoints semánticos (nivel 4) | Función de embeddings (numpy) + `embeddear` | Obligatoria  |
| Agentic (nivel 5) | Implementación en `src/chunk.py` | Obligatoria **(nivel no recomendado)**  |
| Cambio de modelo de embeddings | `config.py`/`.env` | Obligatoria (desde cero)  |
| Cierre definitivo | — | Una única vez, al final  |

**◬ IMPORTANTE:** Si se modifica el modelo de embeddings o los límites del índice: ha de regenerarse el índice desde cero.