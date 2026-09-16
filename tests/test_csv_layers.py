"""
tests/test_csv_layers.py
Tests offline unitarios para csv_advisor.py + csv_transform.py, y para la
integración (load -> advisor -> chunk -> dedup) con las 4 familias reales del
corpus (entidad, grupo de hecho, texto plano, agenda).
Ejecutar:  pytest tests/test_csv_layers.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from langchain_core.documents import Document

from src.chunk import trocear
from src.csv_advisor import (
    CsvAdvice,
    RECETAS_CONOCIDAS,
    advise_csv,
    inspect_csv,
    infer_csv_kind,
    leer_filas_csv,
    match_known_source,
)
from src.csv_transform import transform_csv
from src.load import _load_csv
from src.tsd.dedup import deduplicar

DATA = Path(__file__).resolve().parent.parent / "data"

FICHEROS_REALES: dict[str, str] = {
    "areas": "300390-0-areas-deportivas.csv",
    "abonos": "300085-0-deportes_abonos.csv",
    "descuentos": "300097-0-deportes-descuentos.csv",
    "piscinas": "210227-0-piscinas-publicas.csv",
    "polideportivos": "200186-0-polideportivos.csv",
    "agenda": "212504-0-agenda-actividades-deportes.csv",
}


def _ruta(nombre: str) -> Path:
    ruta = DATA / FICHEROS_REALES[nombre]
    assert ruta.exists()
    return ruta


# ---------------------------------------------------------------------------
# PERFIL: lectura robusta (encoding + cabeceras técnicas)
# ---------------------------------------------------------------------------


class TestPerfil:
    def test_areas_cabeceras_basificadas(self) -> None:
        prof = inspect_csv(str(_ruta("areas")), sample_size=20)
        assert "mxassetnum" in prof.columns
        assert "mxassetnum,c,12" not in " ".join(prof.columns)
        assert prof.n_cols == 14
        assert prof.id_columns == ["mxassetnum"]

    def test_abonos_mesura_real(self) -> None:
        """'nº de abonados' es 100% numérico: entra en measure_columns."""
        prof = inspect_csv(str(_ruta("abonos")), sample_size=500)
        assert "nº de abonados" in prof.columns
        assert "nº de abonados" in prof.measure_columns
        assert prof.numeric_col_ratios["nº de abonados"] >= 0.99

    def test_unico_por_falta(self) -> None:
        """108k filas: 2000 de muestra -> 1948 únicas (0.974); el corpus trae
        ~2,6 % de filas duplicadas reales, no un 100 % de únicos."""
        prof = inspect_csv(str(_ruta("abonos")), sample_size=2000)
        assert prof.uniqueness_ratio >= 0.95

    def test_fila_vacia_no_rompe(self, tmp_path: Path) -> None:
        (tmp_path / "vacio.csv").write_text("a;b\n", encoding="utf-8")
        prof = inspect_csv(str(tmp_path / "vacio.csv"))
        assert prof.n_rows == 0 and prof.n_cols == 2


# ---------------------------------------------------------------------------
# ASSESSMENT: receta conocida + heurística
# ---------------------------------------------------------------------------


class TestAdvisor:
    @pytest.mark.parametrize(
        ("nombre", "kind", "treatment", "dedup"),
        [
            ("areas", "entity_table", "entity_doc", "exact_key"),
            ("abonos", "fact_table", "grouped_doc", "group_only"),
            ("descuentos", "fact_table", "grouped_doc", "group_only"),
            ("piscinas", "entity_table", "entity_doc", "exact_key"),
            ("polideportivos", "entity_table", "entity_doc", "exact_key"),
            ("agenda", "entity_table", "entity_doc", "exact_key"),
        ],
    )
    def test_fuente_conocida(self, nombre: str, kind: str, treatment: str, dedup: str) -> None:
        prof = inspect_csv(str(_ruta(nombre)))
        rec = match_known_source(prof)
        assert rec is not None
        assert rec.csv_kind == kind
        assert rec.treatment == treatment
        assert rec.dedup_policy == dedup

    def test_abonos_claves_reales(self) -> None:
        """Las claves del consejo son columnas reales del CSV (sin acentos)."""
        prof = inspect_csv(str(_ruta("abonos")))
        rec = match_known_source(prof)
        assert rec is not None
        columnas = set(prof.columns)
        for c in rec.grouping_keys:
            assert c in columnas, f"clave inexistente: {c!r}"

    def test_schema_cambiado_degrada_a_heuristica(self, tmp_path: Path) -> None:
        """Fuente conocida pero sin la medida esperada -> heurística, no receta."""
        ruta = _ruta("abonos")
        lineas = ruta.read_text(encoding="latin-1").splitlines()
        cab = lineas[0].split(";")
        i = next(j for j, c in enumerate(cab) if "abonados" in c)
        filas = [
            ";".join(p for j, p in enumerate(l.split(";")) if j != i)
            for l in lineas[1:4]
        ]
        (tmp_path / "mod.csv").write_text(
            ";".join(cab[:i] + cab[i + 1:]) + "\n" + "\n".join(filas) + "\n",
            encoding="latin-1",
        )
        prof = inspect_csv(str(tmp_path / "mod.csv"))
        prof.source = "300085-0-deportes_abonos.csv"
        # La receta exige la medida: aquí falta "nº de abonados" -> heurística
        rec = match_known_source(prof)
        if rec is None:
            consejo = infer_csv_kind(prof)
            assert consejo.csv_kind in ("fact_table", "unknown_csv", "textual_table")
        else:
            pytest.fail("esperaba degradación a heurística")

    def test_heuristica_entidad_por_id(self, tmp_path: Path) -> None:
        p = tmp_path / "entidades.csv"
        p.write_text(
            "ID-NOMBRE;Direccion;Tipo\n" "e1;calle 1;pista\n" "e2;calle 2;rama\n",
            encoding="utf-8",
        )
        prof = inspect_csv(str(p))
        consejo = infer_csv_kind(prof)
        assert consejo.csv_kind == "entity_table"
        assert consejo.dedup_policy == "exact_key"
        assert consejo.id_columns

    def test_heuristica_fallback_unknown(self, tmp_path: Path) -> None:
        p = tmp_path / "sin_estructura.csv"
        p.write_text(
            "a;b;c\n" "1;x;y\n" "2;z;w\n" "3;q;v\n",
            encoding="utf-8",
        )
        prof = inspect_csv(str(p))
        consejo = infer_csv_kind(prof)
        assert consejo.csv_kind == "unknown_csv"
        assert consejo.treatment == "row_as_doc"
        assert consejo.dedup_policy == "exact_only"
        assert consejo.confidence < 0.5

    def test_heuristica_textual(self, tmp_path: Path) -> None:
        p = tmp_path / "narrativo.csv"
        p.write_text(
            "TITULO;DESCRIPCION\n"
            f"{'Texto largo de prueba.' * 30};{'Detalle narrativo.' * 30}\n" * 5,
            encoding="utf-8",
        )
        prof = inspect_csv(str(p))
        consejo = infer_csv_kind(prof)
        assert consejo.csv_kind == "textual_table"
        assert consejo.treatment == "row_as_doc"

    def test_advise_csv_devuelve_pareja(self) -> None:
        ruta = _ruta("piscinas")
        consejo, perfil = advise_csv(str(ruta))
        assert isinstance(consejo, CsvAdvice)
        assert perfil.source == consejo.source


# ---------------------------------------------------------------------------
# TRANSFORM: aplicación de la política
# ---------------------------------------------------------------------------


class TestTransform:
    def test_areas_doc_por_fila_con_clave(self) -> None:
        docs = transform_csv(str(_ruta("areas")))
        assert len(docs) == 84
        d = docs[0]
        assert d.metadata["source"] == "300390-0-areas-deportivas.csv"
        assert d.metadata["csv_kind"] == "entity_table"
        assert d.metadata["dedup_policy"] == "exact_key"
        assert d.metadata["entity_key"]
        assert d.metadata["district"] == "VILLAVERDE"
        assert d.metadata["neighborhood"] == "BUTARQUE"
        assert d.metadata["chunking_hint"] == "no_chunk"
        assert "MXASSETNUM" in d.page_content.upper()

    def test_abonos_agrupado_por_grupo(self) -> None:
        """108k filas -> un documento por clave (mes, centro, tipo)."""
        docs = transform_csv(str(_ruta("abonos")))
        columnas, filas = leer_filas_csv(str(_ruta("abonos")), sample_size=2000)
        claves_esperadas = {
            (f["mes"], f["centro deportivo"], f["tipo de abono"]) for f in filas
        }
        # 1 doc por grupo, con desglose
        assert len(docs) >= len(claves_esperadas)
        assert all(d.metadata["is_aggregated"] for d in docs)
        assert all(d.metadata["dedup_policy"] == "group_only" for d in docs)
        assert all(d.metadata["group_key"] for d in docs)
        d0 = docs[0]
        assert "Resumen del dataset" in d0.page_content
        assert "MUJER" in d0.page_content or "MUJER" in d0.page_content.upper()

    def test_descuentos_agrupado(self) -> None:
        docs = transform_csv(str(_ruta("descuentos")))
        assert len(docs) > 0
        assert all(d.metadata["is_aggregated"] for d in docs)
        assert all(d.metadata["dedup_policy"] == "group_only" for d in docs)

    def test_piscinas_docs_unicos(self) -> None:
        docs = transform_csv(str(_ruta("piscinas")))
        claves = [d.metadata["entity_key"] for d in docs]
        assert len(claves) == len(set(claves)), "cada PK debe dar un doc único"


# ---------------------------------------------------------------------------
# CHUNK: el paso de chunking_hint respeta el hint del advisor
# ---------------------------------------------------------------------------


class TestChunkHint:
    def test_no_chunk_no_trocea(self) -> None:
        d = Document(
            page_content="Instalación deportiva de prueba con descripción amplia. " * 100,
            metadata={"source": "x.csv", "chunking_hint": "no_chunk"},
        )
        chunks = trocear([d], chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_index"] == 0

    def test_light_chunk_corto_pasa_intacto(self) -> None:
        d = Document(
            page_content="Grupo corto",
            metadata={"source": "x.csv", "chunking_hint": "light_chunk"},
        )
        chunks = trocear([d], chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Grupo corto"

    def test_light_chunk_largo_si_trocea(self) -> None:
        texto = "La piscina municipal tiene piscina cubierta. " * 100
        d = Document(
            page_content=texto,
            metadata={"source": "x.csv", "chunking_hint": "light_chunk"},
        )
        chunks = trocear([d], chunk_size=1000)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# DEDUP: la política exacta protege a los chunks de la semántica
# ---------------------------------------------------------------------------


class TestDedupPorPolicy:
    def test_entity_key_protegida_aunque_es_copiada(self) -> None:
        """Dos chunks casi idénticos pero con entity_key distinta: ambos vivan."""
        chunks = [
            Document(
                page_content="Instalación: pista de padel número 1. Horario 8-22h.",
                metadata={
                    "source": "x.csv",
                    "csv_kind": "entity_table",
                    "dedup_policy": "exact_key",
                    "entity_key": "E1",
                    "semantic_score": 0.9,
                },
            ),
            Document(
                page_content="Instalación: pista de padel número 2. Horario 8-22h.",
                metadata={
                    "source": "x.csv",
                    "csv_kind": "entity_table",
                    "dedup_policy": "exact_key",
                    "entity_key": "E2",
                    "semantic_score": 0.9,
                },
            ),
        ]
        embs = [[1.0, 0.0], [0.9999, 0.001]]
        out, _ = deduplicar(chunks, embs)
        assert len(out) == 2, "la dedup semántica no debe tocar claves exactas"

    def test_repetido_literal_descartado(self) -> None:
        chunks = [
            Document(
                page_content="Fila idéntica A",
                metadata={
                    "source": "x.csv",
                    "dedup_policy": "group_only",
                    "group_key": "g1",
                    "chunk_index": 0,
                    "semantic_score": 0.5,
                },
            ),
            Document(
                page_content="Fila idéntica A",
                metadata={
                    "source": "x.csv",
                    "dedup_policy": "group_only",
                    "group_key": "g1",
                    "chunk_index": 0,
                    "semantic_score": 0.5,
                },
            ),
        ]
        out, _ = deduplicar(chunks, [[1.0], [1.0]])
        assert len(out) == 1

    def test_semantico_sigue_funcionando(self) -> None:
        chunks = [
            Document(
                page_content=f"Texto largo número {i} de una serie larga. ",
                metadata={"source": "pdf", "semantic_score": 0.5},
            )
            for i in range(4)
        ]
        embs = [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99]]
        out, _ = deduplicar(chunks, embs, umbral=0.95)
        assert len(out) == 2  # las dos parejas casi idénticas se reducen a una por pareja

    def test_mezcla_politicas(self) -> None:
        chunks = [
            Document(
                page_content="Entidad A: datos A",
                metadata={"source": "x.csv", "dedup_policy": "exact_key",
                          "entity_key": "A", "semantic_score": 0.5},
            ),
            Document(
                page_content="PDF texto semántico corto 1",
                metadata={"source": "y.pdf", "semantic_score": 0.5},
            ),
            Document(
                page_content="PDF texto semántico corto 2 casi igual",
                metadata={"source": "y.pdf", "semantic_score": 0.5},
            ),
        ]
        embs = [[1.0, 0.0], [0.0, 1.0], [0.001, 0.9999]]
        out, _ = deduplicar(chunks, embs, umbral=0.95)
        # entidad intacta + 1 de los dos pdf casi pareados (IP ≈ 0.9999)
        assert len(out) == 2
