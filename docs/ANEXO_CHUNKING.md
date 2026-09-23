# Guía técnica: cambio del nivel de troceado (cómo hacer)

## 1. Objeto

El presente documento describe el procedimiento para modificar el nivel de troceado (o sus parámetros) del pipeline RAG del proyecto, así como los pasos de validación obligatorios tras la modificación. Se redacta como guía operativa de referencia para el equipo.

## 2. Alcance y referencias

- **Aplicabilidad**: el módulo `src/chunk.py` (único punto del corte), el módulo `config.py`, el script `scripts/eval_coherencia_chunks.py` y la prueba `tests/test_chunks.py`.
- **Referencias**: el apartado noveno de `HIGHLIGHTS_CORPUS_DATA.md`, titulado *El troceado con criterio (nivel dos) y su validación*, y el apartado sexto de `FT_CORPUS_DATA.md`, donde se documenta la escala de niveles de troceado (uno: caracteres fijos; dos: recursivo; tres: específico del documento; cuatro: semántico; cinco: agéntico).

## 3. Regla fundamental

**Atención:** cualquier modificación del divisor de texto (el *splitter*) o de las constantes `CHUNK_SIZE` y `CHUNK_OVERLAP` exige la **regeneración completa del índice**. Los vectores almacenados son función directa del texto troceado; un índice no regenerado tras el cambio produce resultados inválidos.

## 4. Estado inicial del sistema

- **El nivel activo** es el dos (recursivo por caracteres), implementado en la función `trocear` del módulo `src/chunk.py`, mediante el divisor recursivo de texto por caracteres y sus separadores de párrafo, salto de línea, frase, espacio y cadena vacía.
- **La validación** corresponde al script `scripts/eval_coherencia_chunks.py` y a la prueba `tests/test_chunks.py`.
- **La indexación** se realiza mediante la función `ejecutar_pipeline`, invocada con la opción de regeneración de índice activada.

### Los niveles disponibles: costes y beneficios

- **El nivel uno (caracteres fijos)**: su coste es muy bajo (sin dependencias, sencillo) y su beneficio es la rapidez extrema, útil como punto de contraste. Indicación: solo como experimento, porque es rígido y corta sin respetar límites de texto.
- **El nivel dos (recursivo por caracteres)**: su coste es bajo (es el nivel actual y no añade dependencias) y su beneficio es el equilibrio entre velocidad y respeto a párrafos y frases. Indicación: es el nivel por defecto del proyecto.
- **El nivel tres (específico del documento)**: su coste es medio (el markdown no añade dependencias, pero las tablas de PDF requieren la librería Unstructured y un pipeline de ingesta distinto) y su beneficio es que preserva la estructura del documento (encabezados, tablas). Indicación: un corpus homogéneo y estructurado.
- **El nivel cuatro (semántico)**: su coste es alto (un lote extra de embeddings por documento fuente; es el nivel más caro) y su beneficio es que trocea por cambio de tema, sin romper unidades de significado. Indicación: únicamente cuando la evaluación lo justifique.
- **El nivel cinco (agéntico)**: su coste es muy alto (lento, caro y no determinista, lo que hace difícil reproducir el índice) y su beneficio es la flexibilidad teórica de corte. Indicación: no se recomienda.

**Preparados en el entorno virtual (comprobado):** los divisores caracter a fijo, recursivo por caracteres, para markdown y para encabezados de markdown, y el basado en spaCy. **No están** instalados el divisor semántico de la librería de divisores de texto ni el divisor semántico; por consiguiente, el nivel cuatro se implementa manualmente con la función `embeddear`.

## 5. Procedimiento

### 5.1. Preparación

1. Se fijará el directorio de trabajo del corpus (el directorio `data/`) y el conjunto de preguntas de evaluación, que deberán permanecer invariantes durante el experimento.
2. Se obtendrá una instantánea de referencia (el número de chunks, el margen de coherencia y la tasa de aciertos) mediante los comandos de la prueba de métricas de troceado y del script de evaluación de coherencia:

    ```powershell
    python -m pytest tests/test_chunks.py::TestChunkMetrics -v -s
    python -m scripts.eval_coherencia_chunks
    ```

3. Se documentará en el informe el valor inicial de las constantes `CHUNK_SIZE` y `CHUNK_OVERLAP` y el divisor empleado (se registra automáticamente al recrear los índices, en el informe `output/informe_index_(...).md` correspondiente).

### 5.2. Aplicación del cambio

Según el objetivo:

- **El ajuste de parámetros (manteniendo el mismo nivel)**: el cambio se hace en el archivo `.env` (global) o por parámetro de la función `trocear` (puntualmente, al llamarla por comando). La comprobación se realiza mediante la comparación de identidad con un valor no nulo, por lo que un solapamiento de cero es un valor válido que respeta el cero.
- **El nivel uno (contraste)**: la sustitución del bloque del divisor en `src/chunk.py` por el divisor de caracteres fijos.
- **El nivel tres (específico del documento)**: la selección del divisor por extensión, manteniendo la firma de la función `trocear` (por ejemplo, el divisor para markdown en los archivos `.md`). Las tablas de PDF requieren la librería Unstructured (no instalada) y un pipeline de ingesta distinto; no se implementa hasta que la evaluación lo pida.
- **El nivel cuatro (semántico)**: la implementación de puntos de corte por similitud coseno entre frases adyacentes, con las funciones de vectorización y la librería numpy; el umbral sugerido va de 0.20 a 0.25 (un umbral más bajo produce más puntos de corte).
- **El nivel cinco (agéntico)**: no se recomienda; únicamente como experimento opcional documentado.

### 5.3. Regeneración del índice

En Python, ejecutar la función del pipeline sobre el directorio del corpus con la opción de regeneración activada:

```python
ejecutar_pipeline("./data", recreate_index=True)
```

Equivalente directo por consola:

```powershell
python main.py ./data --recreate-index
```

### 5.4. Validación (obligatoria)

1. **Coherencia de los cortes:** el margen de coherencia ha de superar 0.05, ya que el solapamiento entre chunks es lo que preserva la continuidad temática del corpus.
2. **Distribución de longitudes:** se documentará en el informe de troceado, dejando constancia de la dispersión de tamaños resultante y su coherencia con los parámetros de corte configurados.
3. **Recuperación (retrieval):** se ejecutará sobre el mismo conjunto de preguntas, con **al menos dos valores de la constante K** (el número de vecinos que se recuperan), dejando constancia de si el mejor acierto conserva su sentido o si, por el contrario, se filtra ruido en los resultados.
4. **La regeneración del índice definitivo.**
5. **La documentación** en `entregables/informe_decisiones.md`: se dejará constancia del nivel empleado y su justificación, del experimento sobre las constantes `CHUNK_SIZE` y `CHUNK_OVERLAP` con sus métricas asociadas, y de al menos un acierto y una abstención observados en la fase de generación.

## 6. Criterio de aceptación del cambio de nivel

La adopción de un nivel superior quedará condicionada a que la evaluación evidencie una mejora medible —ya sea en la tasa de aciertos o en el margen de coherencia— que justifique el coste adicional que conlleva. De no constar dicha mejora, se mantendrá el nivel previo y se documentará la decisión junto con las métricas que la respaldan.

## 7. Errores comunes y su resolución

Se recogen a continuación, en orden, los errores más frecuentes asociados a la modificación del nivel de troceado, junto con la consecuencia que cada uno acarrea y la resolución que corresponde aplicar.

1. **El error**: modificar el divisor (el *splitter*) o las constantes `CHUNK_SIZE` y `CHUNK_OVERLAP` sin la regeneración consiguiente del índice.
   **La consecuencia**: los vectores almacenados quedan obsoletos respecto al nuevo corte, lo que invalida por completo los resultados de recuperación.
   **La resolución**: regenerar el índice de forma íntegra mediante la función del pipeline con la opción de regeneración, o por consola.
2. **El error**: alterar el corpus o el conjunto de preguntas de evaluación durante el desarrollo del experimento.
   **La consecuencia**: las métricas resultantes carecen de comparabilidad entre sí y el experimento pierde su reproducibilidad.
   **La resolución**: fijar el corpus y las preguntas antes de iniciar el experimento y conservar la instantánea de referencia.
3. **El error**: transmitir un solapamiento de cero con la expectativa de que se aplique el valor por defecto.
   **La consecuencia**: confusión sobre el parámetro efectivamente vigente, dado que la comprobación de identidad (comparación con un valor no nulo) respeta expresamente el cero.
   **La resolución**: para aplicar el valor por defecto, omitir el parámetro en lugar de transmitir un cero explícito.
4. **El error**: sustituir el modelo de embeddings sin regenerar el índice.
   **La consecuencia**: el índice resulta incompatible con los nuevos vectores generados.
   **La resolución**: regenerar el índice desde cero.
5. **El error**: sustituir el divisor del nivel tres sin preservar la firma de la función `trocear`.
   **La consecuencia**: se compromete el funcionamiento de los consumidores del módulo, entre ellos el pipeline y la suite de pruebas.
   **La resolución**: seleccionar el divisor según la extensión del archivo dentro de `src/chunk.py`, conservando inalterada la firma de la función `trocear`.
6. **El error**: intentar importar el divisor semántico desde la librería de divisores de texto.
   **La consecuencia**: se produce un error de importación, pues dicho componente no está disponible en el entorno virtual.
   **La resolución**: implementar los puntos de corte de forma manual mediante las funciones de vectorización y la librería numpy (el nivel cuatro).
7. **El error**: fijar el umbral del nivel cuatro fuera del rango recomendado, sin medición previa.
   **La consecuencia**: un umbral excesivamente bajo genera un exceso de puntos de corte; un umbral excesivamente alto fusiona temas distintos.
   **La resolución**: medir el coste y la coherencia antes de fijar el umbral y documentarlo en el informe correspondiente.
8. **El error**: un margen de coherencia inferior a 0.05 tras la modificación.
   **La consecuencia**: el solapamiento resulta insuficiente y se pierde la continuidad temática del corpus.
   **La resolución**: aumentar el solapamiento o reducir el tamaño del chunk, y volver a validar el margen.
9. **El error**: adoptar el nivel cinco (agéntico) como vía principal.
   **La consecuencia**: el índice se vuelve lento, costoso y no reproducible.
   **La resolución**: descartar dicho nivel; su uso queda restringido a un experimento opcional debidamente documentado.
10. **El error**: omitir la documentación del nivel empleado, de los parámetros y de las métricas tras el cambio.
    **La consecuencia**: el experimento pierde reproducibilidad y la memoria técnica queda incompleta.
    **La resolución**: documentar el nivel, los parámetros y las métricas en `entregables/informe_decisiones.md`, conforme al apartado 5.4.
11. **El error**: modificar la lista de separadores del divisor recursivo (por ejemplo, adoptar la lista por defecto).
    **La consecuencia**: los cortes difieren de los del índice vigente y los vectores almacenados quedan obsoletos.
    **La resolución**: regenerar el índice y volver a validar el margen de coherencia.
12. **El error**: fijar el solapamiento igual o superior al tamaño del chunk.
    **La consecuencia**: el divisor lanza un error de valor o produce chunks degenerados.
    **La resolución**: mantener el solapamiento estrictamente inferior al tamaño del chunk y volver a validar el margen y la distribución.
13. **El error**: modificar el tamaño del chunk sin revisar el solapamiento.
    **La consecuencia**: la proporción de solapamiento cambia y puede comprometerse la coherencia temática.
    **La resolución**: recalcular el solapamiento y volver a validar que el margen supere 0.05.
14. **El error**: variar el valor de K entre ejecuciones de la evaluación.
    **La consecuencia**: las métricas de recuperación no resultan comparables entre ejecuciones.
    **La resolución**: fijar al menos dos valores de K en todas las ejecuciones y documentarlos.
15. **El error**: aplicar el divisor para markdown a archivos ajenos a markdown (por ejemplo, archivos PDF).
    **La consecuencia**: se pierde la estructura del documento o se producen cortes incorrectos.
    **La resolución**: seleccionar el divisor según la extensión dentro de la función de troceado; las tablas de PDF requieren la librería Unstructured.
16. **El error**: modificar el script de evaluación de coherencia durante el experimento.
    **La consecuencia**: las métricas dejan de ser comparables con la instantánea de referencia.
    **La resolución**: no alterar el script de evaluación durante el experimento y documentar cualquier modificación posterior.

## 8. Tabla de consulta rápida

- **El ajuste de parámetros (misma estrategia)**: la modificación corresponde a las constantes `CHUNK_SIZE` y `CHUNK_OVERLAP` en el archivo `.env`, o al parámetro de la función `trocear`; la regeneración del índice es obligatoria.
- **El contraste con el nivel uno**: la modificación corresponde al divisor de caracteres fijos en `src/chunk.py`; la regeneración del índice es obligatoria.
- **El cambio de separadores (nivel dos)**: la modificación corresponde a la lista de separadores del divisor recursivo en `src/chunk.py`; la regeneración del índice es obligatoria.
- **La estructura markdown (nivel tres)**: la modificación corresponde al cargador por extensión más el divisor para markdown; la regeneración del índice es obligatoria.
- **Los puntos de corte semánticos (nivel cuatro)**: la modificación corresponde a la función de embeddings (mediante numpy) más la función de vectorización; la regeneración del índice es obligatoria.
- **El nivel agéntico (nivel cinco)**: la modificación corresponde a la implementación en `src/chunk.py`; la regeneración del índice es obligatoria (nivel no recomendado).
- **El cambio del modelo de embeddings**: la modificación corresponde a `config.py` o al archivo `.env`; la regeneración del índice es obligatoria, y se realiza desde cero.
- **El cierre definitivo**: ninguna modificación; el índice se regenera una única vez, al final.

**Importante:** insistir en que, siempre que se modifique el modelo de embeddings o los límites del índice, ha de regenerarse el índice desde cero.
