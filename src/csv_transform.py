"""
src/csv_transform.py
Materializa el `CsvAdvice` del advisor: filas -> Document enriquecidos.

Estrategias (por `treatment`):
  - entity_doc:  una fila = un documento de entidad (dedup exact_key por ID).
                 Para tablas de entidades con ID estable (instalaciones,
                 piscinas, áreas, eventos...).
  - grouped_doc: filas agrupadas por dimensiones (`grouping_keys`) -> un
                 documento por grupo, con el desglose dentro (group_only).
                 Para tablas de hechos repetitivas (abonados, descuentos).
  - row_as_doc:  una fila = un documento plano (fallback y tablas textuales).

El pipeline hereda los metadatos (source, csv_kind, dedup_policy, entity_key,
group_key, group_size, is_aggregated, chunking_hint) porque chunk.py copia la
metadata del documento padre a cada chunk.
"""

from __future__ import annotations

from langchain_core.documents import Document

from .csv_advisor import CsvAdvice, advise_csv, leer_filas_csv

# Columnas con las que se enriquece un documento de entidad (si existen).
_DISTRICT = ("nombre_dis", "distrito")
_NEIGHBORHOOD = ("nombre_bar", "barrio")


def limpiar_valor(valor: object) -> str:
    """Compacta espacios consecutivos (whitespace) de un valor de celda."""
    return " ".join(str(valor or "").split())


def _base_texto(row: dict[str, str], columnas: list[str]) -> str:
    """Texto plano de una fila: 'k: v | k: v | ...' (solo valores no vacíos)."""
    return " | ".join(
        f"{c}: {row[c]}" for c in columnas if row.get(c)
    )


def entidad_a_documento(
    source: str,
    row: dict[str, str],
    row_i: int,
    columnas: list[str],
    advice: CsvAdvice,
) -> Document:
    """Una fila de entidad -> un documento enriquecido (dedup por clave exacta)."""
    entity_id = ""
    for c in advice.id_columns:
        if row.get(c):
            entity_id = row[c]
            break

    extra: dict[str, object] = {}
    for clave, alias in (
        ("district", _DISTRICT),
        ("neighborhood", _NEIGHBORHOOD),
    ):
        for col in alias:
            if row.get(col):
                extra[clave] = row[col]
                break

    metadata: dict[str, object] = {
        "source": source,
        "row": row_i,
        "csv_kind": advice.csv_kind,
        "dedup_policy": advice.dedup_policy,
        "entity_key": entity_id,
        "chunking_hint": "no_chunk",
    }
    metadata.update(extra)
    return Document(page_content=_base_texto(row, columnas), metadata=metadata)


def grupo_a_documento(
    source: str,
    keys: tuple[str, ...],
    rows: list[dict[str, str]],
    columnas: list[str],
    advice: CsvAdvice,
) -> Document:
    """Un grupo de filas de hecho -> un documento agregado (dedup group_only).

    El texto conserva el desglose: una línea por fila con las columnas que no
    son clave de grupo (p. ej. sexo, edad de los abonos).
    """
    claves_set = set(advice.grouping_keys)
    lines = [f"Resumen del dataset {source}."]
    for clave, valor in zip(advice.grouping_keys, keys):
        if valor:
            lines.append(f"{clave}: {valor}")
    lines.append("")
    for row in rows:
        detalle = [f"{c}: {row[c]}" for c in columnas if c not in claves_set and row.get(c)]
        if detalle:
            lines.append("- " + " | ".join(detalle))
    text = "\n".join(lines)

    group_key = "|".join(key for key in keys if key)
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "csv_kind": advice.csv_kind,
            "dedup_policy": advice.dedup_policy,
            "group_key": group_key,
            "group_size": len(rows),
            "is_aggregated": True,
            "chunking_hint": "light_chunk",
        },
    )


def filas_a_documentos(
    source: str, rows: list[dict[str, str]], columnas: list[str], advice: CsvAdvice
) -> list[Document]:
    """Fila por documento (row_as_doc: tablas textuales o unknown_csv)."""
    docs: list[Document] = []
    for i, row in enumerate(rows):
        docs.append(
            Document(
                page_content=_base_texto(row, columnas),
                metadata={
                    "source": source,
                    "row": i,
                    "csv_kind": advice.csv_kind,
                    "dedup_policy": advice.dedup_policy,
                    "chunking_hint": "normal_chunk",
                },
            )
        )
    return docs


def transform_csv(path: str) -> list[Document]:
    """Convierte un CSV en documentos aplicando el consejo del advisor.

    Emite el log `[CSV]` con la decisión (kind, treatment, dedup, confianza),
    listo para el informe de decisiones automáticas.
    """
    advice, _profile = advise_csv(path)
    print(
        f"[CSV]   {advice.source}: {advice.csv_kind} | {advice.treatment} | "
        f"dedup={advice.dedup_policy} | confianza {advice.confidence:.2f} "
        f"({advice.reason})"
    )
    columnas, filas = leer_filas_csv(path)
    source = advice.source

    if advice.treatment == "entity_doc":
        return [
            entidad_a_documento(source, row, i, columnas, advice)
            for i, row in enumerate(filas)
        ]

    if advice.treatment == "grouped_doc" and advice.grouping_keys:
        grupos: dict[tuple[str, ...], list[dict[str, str]]] = {}
        for row in filas:
            key = tuple(row.get(c, "") for c in advice.grouping_keys)
            grupos.setdefault(key, []).append(row)
        return [
            grupo_a_documento(source, key, rows, columnas, advice)
            for key, rows in grupos.items()
        ]

    return filas_a_documentos(source, filas, columnas, advice)
