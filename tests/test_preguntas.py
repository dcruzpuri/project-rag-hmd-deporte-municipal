"""
tests/test_preguntas.py
Validación del corpus de evaluación queries/preguntas.json:
Schema estable (id, pregunta, categoria_esperada, fuente_esperada), el campo
multi-aspecto tags_esperados y reflejo de la taxonomía cerrada de TAG.

Ejecutar:  pytest tests/test_preguntas.py -v
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tsd.tag import CATEGORIAS_VALIDAS, TAGS_VALIDAS

PREGUNTAS = Path(__file__).resolve().parent.parent / "queries" / "preguntas.json"


def _corpus() -> list[dict]:
    return json.loads(PREGUNTAS.read_text(encoding="utf-8"))


def test_dieciocho_pistas_estables() -> None:
    corpus = _corpus()
    assert len(corpus) == 18
    for i, q in enumerate(corpus, start=1):
        assert q["id"] == i  # orden estable: el snapshot de eval no se reordina
        assert isinstance(q["pregunta"], str) and q["pregunta"].strip()
        assert set(q) >= {"id", "pregunta", "categoria_esperada", "fuente_esperada"}


def test_categorias_esperadas_validas() -> None:
    for q in _corpus():
        assert q["categoria_esperada"] in CATEGORIAS_VALIDAS


def test_tags_esperados_presentes_y_validos() -> None:
    """tags_esperados: lista (puede ser vacía), valores ∈ TAGS_VALIDAS, sin repetidos."""
    for q in _corpus():
        tags = q["tags_esperados"]
        assert isinstance(tags, list)
        assert set(tags) <= TAGS_VALIDAS, f"pid {q['id']}: {tags}"
        assert len(tags) == len(set(tags)), f"pid {q['id']}: tags repetidas {tags}"


def test_fuente_esperada_existe_en_data() -> None:
    """Cada fuente_esperada coincide con un fichero real de data/ (basename):
    los renombres del corpus y las preguntas nunca vuelven a desincronizarse."""
    datos = {p.name for p in (Path(__file__).resolve().parent.parent / "data").iterdir()
             if p.is_file()}
    for q in _corpus():
        assert q["fuente_esperada"] in datos, (
            f"pid {q['id']}: {q['fuente_esperada']!r} no existe en data/"
        )


def test_pregunta_multiaspecto_n8() -> None:
    """La nº8 (cancelar con coste / devolución) es multi-aspecto real:
    no está limitada a una sola etiqueta (la categoría sigue siendo única)."""
    q = next(x for x in _corpus() if x["id"] == 8)
    assert len(q["tags_esperados"]) >= 2
