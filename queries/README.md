# Preguntas de evaluación

Dataset de evaluación del RAG de Deporte Municipal Madrid. En total son 20 preguntas: 18 sobre contenido del corpus y 2 fuera de dominio para comprobar que el sistema sabe cuándo no tiene información.

## Archivo

`evaluation_questions.json` con 20 preguntas organizadas por temática.

## Estructura

Cada pregunta tiene los siguientes campos:

- `id` — identificador único
- `pregunta` — pregunta en lenguaje natural
- `categoria_esperada` — categoría temática esperada
- `fuente_esperada` — documento del corpus del que debería venir la respuesta
- `out_of_corpus` — indica si la pregunta está fuera del corpus (solo en preguntas out-of-corpus)

## Categorías

| Categoría | Preguntas |
|---|---|
| tarifas | 1, 2, 3 |
| abonos | 4, 5, 6 |
| reservas | 7, 8, 9 |
| normativa | 10, 11, 12 |
| instalaciones | 13, 14, 15 |
| agenda | 16, 17, 18 |
| out_of_corpus | 19, 20 |

## Para qué se usan

Las preguntas se usan para evaluar:

- **Retrieval** — ¿recupera el chunk correcto? Probado con al menos 2 valores de TOP_K
- **Generación** — ¿la respuesta está anclada al corpus?
- **Abstención** — preguntas 19 y 20 deben devolver "no está en los documentos"