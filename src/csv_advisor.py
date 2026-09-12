"""
src/csv_advisor.py

Advisor de CSV (datos abiertos). Evalúa un archivo CSV **a partir de su
esquema real** y deriva su plan de carga, sin mapeo manual por nombre de
archivo. Un CSV nuevo (fuente desconocida) se evalúa, se adapta a un
tratamiento adecuado y se emite su plan para que sea confirmado.

Estrategias:
  - ``entity`` : catálogo. Una fila por entidad, identificada por una columna ID
                 (p. ej. MXASSETNUM). Se aplica dedup exacta por ese ID.
  - ``grupos`` : agregable. Se agrupan las filas por una combinación de claves
                 lógicas y se conserva el desglose dentro (p. ej. abonos y
                 descuentos). Se aplica dedup exacta por clave de grupo.
  - ``fila``   : hecho ancho. Una fila por registro (``campo: valor``). Dedup
                 exacta por contenido de fila.

La detección es heurística (esquema + valores). Puede fijarse un plan manual
como override para reproducir un tratamiento confirmado.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Utilidades de lectura (compartidas con load.py)
# ---------------------------------------------------------------------------

# En orden de preferencia. cp1252 mapea mejor que latin-1 los guiones y
# comillas típicas de Windows (0x95, 0x96, 0x92...), pero algunos bytes
# (0x81, 0x8D, 0x8F, 0x90, 0x9D) no existen en cp1252: ahí entra latin-1.
_CODIFICACIONES: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")


def detectar_encoding(path: str) -> str:
    """Detecta la encoding probando utf-8-sig, cp1252 y latin-1 (nunca falla)."""
    raw = Path(path).read_bytes()
    for enc in _CODIFICACIONES:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detectar_delimitador(path: str) -> str:
    """
    Detecta el separador de un CSV probando ``;``, ``,`` y ``\t``.

    Criterio de consistencia: gana el candidato cuyo número de columnas es
    **estable entre la cabecera y las primeras filas de datos**. Es necesario
    porque una cabecera técnica (p. ej. ``MXASSETNUM,C,12;DESCRIPCIO,C,105``)
    contiene más comas que separadores en su propia primera línea, así que el
    conteo simple en la cabecera fallaría.

    Returns:
        El separador detectado (``";"`` por defecto).
    """
    import io

    encoding = detectar_encoding(path)
    texto = Path(path).read_text(encoding=encoding)
    # Las primeras 5 líneas bastan: cabecera + 4 filas de datos.
    lineas = [l for l in texto.splitlines() if l.strip()][:5]
    mejor: str = ";"
    mejor_cols: int = 0
    for sep in (";", ",", "\t"):
        conteos = [
            len(fila)
            for fila in csv.reader(io.StringIO("\n".join(lineas)), delimiter=sep)
            if fila
        ]
        if not conteos:
            continue
        # Separador "bueno": nº de columnas estable entre cabecera y datos y
        # no degenerado (>1 columnas). Con el incorrecto, la cabecera y los
        # datos difieren (o todo queda en 1 columna, el caso vacío).
        if conteos.count(conteos[0]) == len(conteos) and conteos[0] > 1:
            # A más columnas, más estructurado: el separador que más divide
            # (sin romper) es el correcto.
            if conteos[0] >= mejor_cols:
                mejor, mejor_cols = sep, conteos[0]
    return mejor


def _iter_filas(path: str) -> list[dict[str, str]]:
    """Lee todas las filas de un CSV como dict normalizado (sin claves None)."""
    encoding = detectar_encoding(path)
    delimitador = detectar_delimitador(path)
    with open(path, newline="", encoding=encoding) as f:
        raw = f.read()
    filas: list[dict[str, str]] = []
    for row in csv.DictReader(raw.splitlines(), delimiter=delimitador):
        filas.append(
            {
                str(k).strip(): (v or "").strip()
                for k, v in row.items()
                if k is not None and v is not None and v
            }
        )
    return filas


# ---------------------------------------------------------------------------
# Perfilado de columnas (heurísticas sobre el esquema)
# ---------------------------------------------------------------------------


def _base(c: str) -> str:
    """Nombre de columna sin el sufijo técnico técnico: 'MXASSETNUM,C,12' -> 'MXASSETNUM'."""
    return (c.split(",")[0]).strip() if c else c


# Métrica de cantidad (se agrega por suma o conteo dentro del grupo).
# Solo se aplica a columnas "numéricas" (lo comprueba el perfil, no el nombre).
_RE_METRICA = re.compile(
    r"(?i)(abonado|descuento|recargo|total|cantidad|precio|importe|valor"
    r"|^num|\bn§\b|\bnº\b|\bnum\b|\bcount\b)"
)
# Detalle: columnas que se conservan DENTRO del grupo (no definen la clave).
_RE_DETALLE = re.compile(r"(?i)(\bsexo\b|\bedad\b|[g]énero|genero|actividad|gasto)")


@dataclass
class _Perfil:
    """Perfiles mínimo de una columna para decidir su papel."""

    numeric_ratio: float = 0.0
    empty_ratio: float = 0.0
    unique_ratio: float = 0.0
    n_unicos: int = 0


def _perfilar(filas: list[dict[str, str]], c: str) -> _Perfil:
    """Calcula ratios (numérico/vacío/único) de la columna ``c``."""
    n = len(filas)
    if not n:
        return _Perfil()
    n_num = 0
    n_empty = 0
    vistos: set[str] = set()
    for f in filas:
        v = f.get(c, "")
        if not v:
            n_empty += 1
            continue
        vistos.add(v)
        try:
            float(v)
            n_num += 1
        except ValueError:
            pass
    return _Perfil(
        numeric_ratio=n_num / n,
        empty_ratio=n_empty / n,
        unique_ratio=len(vistos) / n,
        n_unicos=len(vistos),
    )


def _es_metrica(c: str) -> bool:
    return bool(_RE_METRICA.search(_base(c)))


def _es_detalle(c: str) -> bool:
    return bool(_RE_DETALLE.search(_base(c)))


# Nº de columnas por debajo de las cuales un fichero con ID se trata como
# catálogo (entity) en vez de hecho ancho (fila).
_MAX_COLS_ENTITY = 20


@dataclass
class PlanCsv:
    """Plan de carga derivado por el advisor para un CSV concreto."""

    path: str = ""
    estrategia: str = "fila"  # "entity" | "grupos" | "fila"
    delimitador: str = ";"
    encoding: str = "latin-1"
    n_filas: int = 0
    # entity
    id_column: str | None = None
    label_column: str | None = None
    # grupos
    metric_column: str | None = None
    group_keys: tuple[str, ...] = ()
    detail_columns: tuple[str, ...] = ()
    atomic: bool = True  # True: cada fila=1 unidad (contar); False: sumar.
    # fila
    columns: tuple[str, ...] = ()
    # diagnóstico
    ids_unicos: int = 0
    n_ids_repetidos: int = 0
    es_nuevo: bool = True


class CsvAdvisor:
    """Evalúa un CSV y devuelve su :class:`PlanCsv`.

    Args:
        overrides: planes fijados por ``source`` (nombre de archivo). Un CSV con
            override se sirve sin re-inferir (reproducible y barato).
        known_sources: fuentes ya evaluadas en pasadas anteriores. Un archivo no
            en este conjunto y sin override se marca ``es_nuevo`` (para emitir
            su plan recién detectado).
    """

    def __init__(
        self,
        overrides: dict[str, PlanCsv] | None = None,
        known_sources: set[str] | None = None,
    ) -> None:
        self._overrides = overrides or {}
        self._known = known_sources or set()

    def plan(self, path: str) -> PlanCsv:
        """Devuelve el plan (fijado o inferido) para ``path``."""
        source = Path(path).name
        if source in self._overrides:
            return dataclasses.replace(
                self._overrides[source], path=path, es_nuevo=False
            )
        plan = self._infer(path)
        plan.es_nuevo = source not in self._known
        return plan

    # ------------------------------------------------------------------ #
    def _infer(self, path: str) -> PlanCsv:
        filas = _iter_filas(path)
        plan = PlanCsv(path=path)
        plan.delimitador = detectar_delimitador(path)
        plan.encoding = detectar_encoding(path)
        plan.n_filas = len(filas)
        if not filas:
            return plan

        columnas = list(filas[0].keys())
        plan.columns = tuple(columnas)
        perfil = {c: _perfilar(filas, c) for c in columnas}

        id_column = self._detectar_id(filas, perfil)
        plan.id_column = id_column
        if id_column is not None:
            plan.ids_unicos = perfil[id_column].n_unicos
            plan.n_ids_repetidos = len(filas) - plan.ids_unicos
        metric_column = self._detectar_metrica(columnas, perfil, id_column)
        plan.metric_column = metric_column

        if id_column is not None:
            plan.estrategia = (
                "entity" if len(columnas) <= _MAX_COLS_ENTITY else "fila"
            )
            if plan.estrategia == "entity":
                plan.label_column = self._elegir_label(columnas, perfil, id_column)
        elif metric_column is not None:
            plan.estrategia = "grupos"
            detalle = tuple(c for c in columnas if _es_detalle(_base(c)))
            claves = tuple(
                c for c in columnas
                if c != metric_column and c != id_column and c not in detalle
                and _base(c)
            )
            plan.detail_columns = detalle
            plan.group_keys = claves
            plan.atomic = self._es_atomico(filas, metric_column)
        else:
            plan.estrategia = "fila"

        return plan

    # ------------------------------------------------------------------ #
    @staticmethod
    def _detectar_id(filas: list[dict[str, str]], perfil: dict[str, _Perfil]) -> str | None:
        """Primer candidata a ID: único (~100 %) y no-métrica."""
        for c, p in perfil.items():
            if p.unique_ratio >= 0.9 and p.empty_ratio <= 0.05 and p.n_unicos >= 2:
                if not _RE_METRICA.search(_base(c)):
                    return c
        return None

    @staticmethod
    def _detectar_metrica(
        columnas: list[str], perfil: dict[str, _Perfil], id_column: str | None
    ) -> str | None:
        for c in columnas:
            if c == id_column:
                continue
            p = perfil[c]
            if p.numeric_ratio >= 0.8 and _RE_METRICA.search(_base(c)):
                return c
        return None

    @staticmethod
    def _es_atomico(filas: list[dict[str, str]], metric: str) -> bool:
        """True si la métrica casi siempre es 1 (cada fila = 1 unidad)."""
        valores: list[int] = []
        for f in filas:
            try:
                valores.append(int(float(f.get(metric, ""))))
            except ValueError:
                pass
        if not valores:
            return True
        return sum(1 for v in valores if v == 1) / len(valores) >= 0.9

    @staticmethod
    def _elegir_label(
        columnas: list[str], perfil: dict[str, _Perfil], id_column: str | None
    ) -> str | None:
        """Columna de texto más descriptiva (no numérica, no vacía, no ID)."""
        for c in columnas:
            if c == id_column or not c:
                continue
            if perfil[c].numeric_ratio < 0.5 and perfil[c].empty_ratio < 0.9:
                return c
        return None


# ---------------------------------------------------------------------------
# Serialización del plan (para fijar/confirmar planes)
# ---------------------------------------------------------------------------


def plan_a_dict(plan: PlanCsv) -> dict:
    """Convierte un PlanCsv en dict serializable (sin ``path``/``es_nuevo``)."""
    d = asdict(plan)
    d.pop("path", None)
    d.pop("es_nuevo", None)
    return d


def dict_a_plan(data: dict, path: str) -> PlanCsv:
    """Reconstruye un PlanCsv desde un dict (usado por override)."""
    campos = {
        k: v for k, v in data.items()
        if k in {
            "estrategia", "delimitador", "encoding", "id_column", "label_column",
            "metric_column", "atomic", "n_filas",
        }
    }
    return PlanCsv(
        path=path,
        estrategia=campos.get("estrategia", "fila"),
        delimitador=campos.get("delimitador", ";"),
        encoding=campos.get("encoding", "latin-1"),
        id_column=campos.get("id_column"),
        label_column=campos.get("label_column"),
        metric_column=campos.get("metric_column"),
        atomic=campos.get("atomic", True),
        n_filas=campos.get("n_filas", 0),
        es_nuevo=False,
    )
