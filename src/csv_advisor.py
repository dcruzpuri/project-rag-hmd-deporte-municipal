"""
src/csv_advisor.py
Capa de clasificación + política para CSV.

El corpus trae CSVs muy heterogéneos (entidades con ID estable, tablas de
hechos repetitivas, ficheros textuales): tratarlos todos como "una fila =
un documento" fuerza a todos el mismo chunking y la misma deduplicación
semántica. Este módulo decide el tratamiento ANTES de trocear y deduplicar:

    LOAD -> CSV_ADVISOR (perfil estructural) -> [CSV_TRANSFORM]
         -> CLEAN -> CHUNK -> EMBED -> SCORING/DEDUP (por dedup_policy) -> INDEX

Dos niveles de decisión (lo barato primero, sin embeddings nunca):
  1. `match_known_source`: fuente conocida de datos.madrid.es por `source`
     (receta fija; valida que las columnas esperadas siguen presentes).
  2. `infer_csv_kind`: heurística sobre el perfil estructural (cardinalidades,
     densidad numérica, repetición, columnas ID/medida/temporal/narrativa).

Un CSV nuevo y desconocido cae al conservador `unknown_csv` / `exact_only`:
no rompe nada y queda registrado para revisar.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Lectura robusta (compartida con load.py): encoding + delimitador
# ---------------------------------------------------------------------------

_CODIFICACIONES: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")
_DELIMITADORES: tuple[str, ...] = (";", ",", "\t")

# Detección por nombre de columna (heuristicas del enunciado, no del fichero).
_ID_TOKENS: tuple[str, ...] = ("assetnum", "asset", "pk", "codigo", "id")
_MEDIDA_TOKENS: tuple[str, ...] = (
    "abonado", "descuento", "recargo", "precio", "importe",
    "total", "cantidad", "valor",
)
_TEMPORAL_TOKENS: tuple[str, ...] = ("fecha", "periodo")
_NARRATIVA_TOKENS: tuple[str, ...] = (
    "descripcion", "descrip", "horario", "equipamiento", "trans",
    "audiencia", "texto", "url", "titulo", "contenido", "estado",
)


def detectar_encoding(path: str) -> str:
    """Encoding del fichero: utf-8-sig -> cp1252 -> latin-1 (latin-1 nunca falla)."""
    raw = Path(path).read_bytes()
    for enc in _CODIFICACIONES:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


# La º ordinal del corpus sale corrompida según el encoding (0xA7 latin-1/cp1252
# -> §): se normaliza a º para que las recetas enganchen sin importar el byte.
_ORDINAL_ALIASES: dict[str, str] = {"§": "º"}


def normalizar_col(nombre: str) -> str:
    """'MXASSETNUM,C,12' -> 'mxassetnum' (cabeceras técnicas de datos.madrid.es)."""
    base = (nombre or "").split(",")[0].strip().lower()
    return "".join(_ORDINAL_ALIASES.get(ch, ch) for ch in base)


def detectar_delimitador(path: str) -> str:
    """Sep. cuyo nº de columnas es estable entre cabecera y datos (y no degenera a 1)."""
    lineas = [
        l for l in Path(path).read_text(encoding=detectar_encoding(path)).splitlines()
        if l.strip()
    ][:5]
    mejor: str = ";"
    mejor_cols: int = 0
    for sep in _DELIMITADORES:
        conteos = [
            len(fila)
            for fila in csv.reader(io.StringIO("\n".join(lineas)), delimiter=sep)
            if fila
        ]
        if not conteos:
            continue
        if conteos.count(conteos[0]) == len(conteos) and conteos[0] > 1:
            if conteos[0] >= mejor_cols:
                mejor, mejor_cols = sep, conteos[0]
    return mejor


def leer_filas_csv(path: str, sample_size: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    """Lee un CSV como dict normalizado: claves basificadas (minúscula, sin sufijo técnico).

    Args:
        path: ruta del CSV.
        sample_size: límite de filas leídas (para el perfil); `None` = todo.
    """
    filas: list[dict[str, str]] = []
    columnas: list[str] = []
    encoding = detectar_encoding(path)
    delimitador = detectar_delimitador(path)
    with open(path, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimitador)
        for c in reader.fieldnames or []:
            base = normalizar_col(c)
            if base not in columnas:
                columnas.append(base)
        for i, row in enumerate(reader):
            limpio: dict[str, str] = {}
            for raw_k, raw_v in row.items():
                if raw_k is None:  # fila más larga que la cabecera (ruido)
                    columnas.append("col")
                    continue
                base = normalizar_col(raw_k)
                limpio[base] = (raw_v or "").strip()
                if base not in columnas:
                    columnas.append(base)
            filas.append(limpio)
            if sample_size is not None and i + 1 >= sample_size:
                break
    return list(dict.fromkeys(columnas)), filas


# ---------------------------------------------------------------------------
# Perfil estructural
# ---------------------------------------------------------------------------


@dataclass
class CsvProfile:
    """Métricas estructurales de un CSV (sobre la muestra leída)."""

    source: str
    path: str
    columns: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    id_columns: list[str] = field(default_factory=list)
    measure_columns: list[str] = field(default_factory=list)
    temporal_columns: list[str] = field(default_factory=list)
    narrative_columns: list[str] = field(default_factory=list)
    numeric_ratio: float = 0.0  # prop. de columnas con ≥50 % valores numéricos
    narrative_ratio: float = 0.0
    avg_row_chars: float = 0.0
    uniqueness_ratio: float = 0.0  # 1.0 = todas las filas distintas
    has_date_column: bool = False
    # Internos de inferencia (no se serializan)
    numeric_col_ratios: dict[str, float] = field(default_factory=dict)
    top_cardinality: dict[str, int] = field(default_factory=dict)


def _parece_id(columna: str) -> bool:
    c = normalizar_col(columna)
    return any(tok in c for tok in _ID_TOKENS)


def _parece_medida(columna: str) -> bool:
    c = normalizar_col(columna)
    return any(tok in c for tok in _MEDIDA_TOKENS)


def _parece_temporal(columna: str) -> bool:
    c = normalizar_col(columna)
    return any(tok in c for tok in _TEMPORAL_TOKENS)


def _parece_narrativa(columna: str) -> bool:
    c = normalizar_col(columna)
    return any(tok in c for tok in _NARRATIVA_TOKENS)


def _parece_num(valor: str) -> bool:
    texto = valor.replace(" ", "").replace(",", ".")
    try:
        float(texto)
        return True
    except ValueError:
        return False


def inspect_csv(path: str, sample_size: int = 2000) -> CsvProfile:
    """Perfil estructural de un CSV (muestra leída; nunca lee el fichero entero).

    Args:
        path: ruta del CSV.
        sample_size: nº de filas de datos perfiladas (1000-5000 es suficiente).
    """
    columnas, filas = leer_filas_csv(path, sample_size=sample_size)
    n_rows = len(filas)
    n_cols = len(columnas)

    id_columns = [c for c in columnas if _parece_id(c)]
    temporal_columns = [c for c in columnas if _parece_temporal(c)]
    measure_nombradas = [c for c in columnas if _parece_medida(c)]
    narrative = [c for c in columnas if _parece_narrativa(c)]

    # Densidad numérica real de cada columna (sobre las primeras 1000 filas no vacías).
    numeric_ratios: dict[str, float] = {}
    cardinalidad: dict[str, int] = {}
    for c in columnas:
        valores = [r[c] for r in filas[:1000] if r.get(c)]
        if not valores:
            continue
        numeric_ratios[c] = sum(1 for v in valores if _parece_num(v)) / len(valores)
        cardinalidad[c] = len(set(valores))

    num_cols = sum(1 for r in numeric_ratios.values() if r >= 0.5)
    num_ratio = num_cols / len(numeric_ratios) if numeric_ratios else 0.0
    narr_ratio = len(narrative) / n_cols if n_cols else 0.0

    filas_str = ["|".join(r.get(c, "") for c in columnas) for r in filas]
    uniqueness = len(set(filas_str)) / n_rows if n_rows else 0.0
    avg_chars = sum(len(s) for s in filas_str) / n_rows if n_rows else 0.0

    # Medidas reales: la densidad numérica decide (una columna "precio" vacía
    # no es una medida para este corpus).
    medidas_efectivas = [
        c for c in measure_nombradas if numeric_ratios.get(c, 0.0) >= 0.5
    ]

    return CsvProfile(
        source=Path(path).name,
        path=path,
        columns=columnas,
        n_rows=n_rows,
        n_cols=n_cols,
        id_columns=id_columns,
        measure_columns=medidas_efectivas,
        temporal_columns=temporal_columns,
        narrative_columns=narrative,
        numeric_ratio=round(num_ratio, 3),
        narrative_ratio=round(narr_ratio, 3),
        avg_row_chars=round(avg_chars, 1),
        uniqueness_ratio=round(uniqueness, 3),
        has_date_column=bool(temporal_columns),
        numeric_col_ratios=numeric_ratios,
        top_cardinality=cardinalidad,
    )


# ---------------------------------------------------------------------------
# Consejo ejecutable (la política que el pipeline aplica)
# ---------------------------------------------------------------------------


@dataclass
class CsvAdvice:
    """Decisión de tratamiento para un CSV.

    Fields:
        csv_kind: `entity_table` | `fact_table` | `time_series` |
            `textual_table` | `unknown_csv`.
        treatment: `entity_doc` | `grouped_doc` | `row_as_doc`.
        dedup_policy: `exact_only` | `exact_key` | `group_only` |
            `semantic_optional` (los demás por omisión -> dedup semántica).
        id_columns / measure_columns / grouping_keys: en forma basificada
            (la misma que las claves de las filas de `leer_filas_csv`).
    """

    source: str
    csv_kind: str
    treatment: str
    dedup_policy: str
    id_columns: list[str] = field(default_factory=list)
    measure_columns: list[str] = field(default_factory=list)
    grouping_keys: list[str] = field(default_factory=list)
    confidence: float = 0.5
    reason: str = ""


# Recetas de fuentes conocidas (datos.madrid.es). Las claves van base-encoded
# (mismo formato que `leer_filas_csv`). Si la regeneración cambia el esquema,
# `match_known_source` devuelve `None` y la heurística toma el relevo.
RECETAS_CONOCIDAS: dict[str, dict[str, object]] = {
    "200186-0-polideportivos.csv": {
        "kind": "entity_table", "treatment": "entity_doc", "dedup": "exact_key",
        "id_columns": ["pk"],
        "reason": "entidades (centros) con PK estable",
    },
    "200215-0-instalaciones-deportivas.csv": {
        "kind": "entity_table", "treatment": "entity_doc", "dedup": "exact_key",
        "id_columns": ["pk"],
        "reason": "entidades (instalaciones) con PK estable",
    },
    "210227-0-piscinas-publicas.csv": {
        "kind": "entity_table", "treatment": "entity_doc", "dedup": "exact_key",
        "id_columns": ["pk"],
        "reason": "entidades (piscinas) con PK estable",
    },
    "300390-0-areas-deportivas.csv": {
        "kind": "entity_table", "treatment": "entity_doc", "dedup": "exact_key",
        "id_columns": ["mxassetnum"],
        "reason": "entidades (áreas) con identificador estable",
    },
    "300085-0-deportes_abonos.csv": {
        "kind": "fact_table", "treatment": "grouped_doc", "dedup": "group_only",
        "id_columns": [], "claves": ["mes", "centro deportivo", "tipo de abono"],
        "medidas": ["nº de abonados"],
        "reason": "tabla de hechos masiva; 1 fila = 1 abonado",
    },
    "300097-0-deportes-descuentos.csv": {
        "kind": "fact_table", "treatment": "grouped_doc", "dedup": "group_only",
        "id_columns": [], "claves": ["grupo descuento/recargo", "mes",
                                       "centro deportivo", "distrito"],
        "medidas": ["nº descuentos"],
        "reason": "tabla de hechos; 1 fila = 1 descuento",
    },
    "212504-0-agenda-actividades-deportes.csv": {
        "kind": "entity_table", "treatment": "entity_doc", "dedup": "exact_key",
        "id_columns": ["id-evento"],
        "reason": "eventos con identificador estable",
    },
}


def match_known_source(profile: CsvProfile) -> CsvAdvice | None:
    """Receta fija para `source` conocido. `None` si la fuente no es
    conocida o su esquema ya no contiene las columnas esperadas."""
    receta = RECETAS_CONOCIDAS.get(profile.source)
    if receta is None:
        return None
    claves = [c for c in (receta.get("claves") or ()) if c not in profile.columns]
    ids = [c for c in (receta.get("id_columns") or ()) if c not in profile.columns]
    medidas = [c for c in (receta.get("medidas") or ()) if c not in profile.columns]
    if claves or ids or medidas:  # esquema cambiado: degradar a heurística
        return None
    return CsvAdvice(
        source=profile.source,
        csv_kind=str(receta["kind"]),
        treatment=str(receta["treatment"]),
        dedup_policy=str(receta["dedup"]),
        id_columns=list(receta.get("id_columns") or ()),
        measure_columns=list(receta.get("medidas") or ()),
        grouping_keys=list(receta.get("claves") or ()),
        confidence=0.99,
        reason=str(receta["reason"]),
    )


def _elegir_claves(profile: CsvProfile) -> list[str]:
    """Claves de agrupación para un unknown fact: dimensiones (sin medidas,
    sin IDs, sin narrativas), temporales primero, máx. 4."""
    medidas = set(profile.measure_columns)
    ids = set(profile.id_columns)
    dimensiones: list[str] = []
    for c in profile.temporal_columns:
        if c not in medidas and c not in ids:
            dimensiones.append(c)
    for c in profile.columns:
        if (c not in medidas and c not in ids and c not in dimensiones
                and not _parece_narrativa(c)):
            dimensiones.append(c)
        if len(dimensiones) >= 4:
            break
    return dimensiones[:4]


def infer_csv_kind(profile: CsvProfile) -> CsvAdvice:
    """Heurística por perfil estructural (CSV nuevo o fuente renombrada).

    Orden (de más específica a más genérica):
       1. entity: columna ID (catalogo), sin periodo temporal
          -> entity_doc + exact_key (la dedup semántica rompe IDs casi iguales).
      2. fact/time_series: medidas numéricas reales + 3+ dimensiones
         -> grouped_doc + group_only (agrupar por dimensiones; las filas
            repetitivas no compiten en dedup semántica).
      3. textual: filas ricas en texto o columnas narrativas dominantes
         -> row_as_doc + semantic_optional.
      4. unknown: fila por documento + exact_only (fallback seguro).
    """
    n_cols = profile.n_cols

    def _claves(
        kind: str,
        treatment: str,
        dedup: str,
        conf: float,
        razon: str,
        claves: list[str] | None = None,
        medidas: list[str] | None = None,
        ids: list[str] | None = None,
    ) -> CsvAdvice:
        return CsvAdvice(
            source=profile.source,
            csv_kind=kind,
            treatment=treatment,
            dedup_policy=dedup,
            id_columns=ids if ids is not None else profile.id_columns[:1],
            measure_columns=medidas if medidas is not None else profile.measure_columns[:1],
            grouping_keys=claves if claves is not None else [],
            confidence=conf,
            reason=razon,
        )

    # 1. Entidades.
    if profile.id_columns and n_cols >= 3 and not profile.temporal_columns:
        return _claves(
            "entity_table", "entity_doc", "exact_key", 0.8,
            "ID estable en tabla de entidad -> dedup exacta por clave",
            ids=profile.id_columns[:2],
        )

    # 2. Tabla de hechos.
    dimensiones = [
        c for c in profile.columns
        if c not in profile.measure_columns
        and c not in profile.id_columns
        and not _parece_narrativa(c)
    ]
    narrativas = len(profile.narrative_columns)
    narr_ratio = narrativas / n_cols if n_cols else 0.0
    if (
        profile.measure_columns
        and len(dimensiones) >= 3
        and narr_ratio < 0.3
    ):
        claves = _elegir_claves(profile)
        kind = (
            "time_series"
            if profile.has_date_column and len(profile.measure_columns) >= 1
            else "fact_table"
        )
        conf = 0.85 if kind == "time_series" else 0.75
        return _claves(
            kind, "grouped_doc", "group_only", conf,
            "medidas + dimensiones -> agrupar por dimensiones (group_only)",
            claves=claves,
        )

    # 3. Tabla textual.
    if (
        profile.avg_row_chars > 250
        or narrativas >= 2
        or narr_ratio >= 0.5
    ):
        return _claves(
            "textual_table", "row_as_doc", "semantic_optional", 0.65,
            "filas textuales: cada fila es un documento (dedup semántica opcional)",
        )

    # 4. Fallback conservador.
    return _claves(
        "unknown_csv", "row_as_doc", "exact_only", 0.4,
        "CSV no reconocido: fila por documento, dedup solo exacta",
        medidas=profile.measure_columns[:1],
    )


def advise_csv(path: str) -> tuple[CsvAdvice, CsvProfile]:
    """Clasifica un CSV: receta de fuente conocida o heurística de perfil.

    Returns:
        (consejo ejecutable, perfil estructural) para el log de `[CSV]`.
    """
    profile = inspect_csv(path)
    consejo = match_known_source(profile)
    if consejo is None:
        consejo = infer_csv_kind(profile)
    return consejo, profile
