# Preguntas de evaluación

Dataset de evaluación del RAG de Deporte Municipal Madrid. En total son 22 preguntas: 18 sobre contenido del corpus y 4 fuera de dominio intercaladas para comprobar que el sistema sabe cuándo no tiene información.

## Archivo

`evaluation_questions.json` con 22 preguntas organizadas por temática.

## Estructura

Cada pregunta tiene:

- `id` — número
- `pregunta` — la pregunta tal cual
- `categoria_esperada` — de qué va la pregunta
- `fuente_esperada` — de qué archivo debería salir la respuesta
- `out_of_corpus` — solo en las preguntas fuera de dominio

## Distribución

| Categoría | IDs |
|---|---|
| tarifas | 1, 2, 3 |
| abonos | 5, 6, 7 |
| reservas | 8, 10, 11 |
| normativa | 12, 13, 15 |
| instalaciones | 16, 17, 18 |
| agenda | 19, 20, 21 |
| out_of_corpus | 4, 9, 14, 22 |

## Para qué se usan

Para comprobar tres cosas: si el retrieval devuelve los chunks correctos (con al menos dos valores de TOP_K), si la respuesta generada viene del corpus y no se inventa nada, y si el sistema se abstiene cuando la pregunta no tiene respuesta en los documentos — eso lo cubren las preguntas 4, 9, 14 y 22, que están intercaladas a propósito para que no sean predecibles.