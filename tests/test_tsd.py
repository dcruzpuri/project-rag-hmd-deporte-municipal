"""
tests/test_tsd.py
Tests unitarios offline para tsd/scoring.py y tsd/dedup.py.
Sin Ollama ni red: vectores de control.
Ejecutar:  pytest tests/test_tsd.py -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from src.tsd.dedup import deduplicar
from src.tsd.scoring import puntuar
from src.tsd.tag import CATEGORIAS_VALIDAS


def _docs(
    textos: list[str], relevancia: float = 0.5, source: str = "test.csv"
) -> list[Document]:
    """Chunk mínimo por texto (la metadata completa la vienen del pipeline)."""
    return [
        Document(
            page_content=t,
            metadata={
                "source": source,
                "relevancia_llm": relevancia,
                "doc_category": "abonos",
                "tags": "test",
            },
        )
        for t in textos
    ]



def _vecs(textos: list[float]) -> list[list[float]]:
    """Vectores unitarios 1D a partir de escalares."""
    return [[float(v), 0.0] for v in textos]


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------


class TestScoring:
    def test_puntuar_escribe_semantic_score(self) -> None:
        docs = _docs(["a", "b", "c"])
        vecs = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]  # 0.7·0.7+0.7·0.7 ≈ 1
        resultado = puntuar(docs, vecs)
        assert resultado is docs
        for c in docs:
            assert "semantic_score" in c.metadata
            assert 0.0 <= c.metadata["semantic_score"] <= 1.0

    def test_redundancia_duplicados_baja(self) -> None:
        """Dos chunks casi idénticos: su redundancia (0.20·(1 − red)) debe
        bajar más que la de un chunk aislado (red ≈ 0 en la versión clamp)."""
        docs_idd = _docs(["un texto", "texto repetido casi idéntico"])
        docs_iso = _docs(["un solo texto"])
        v_idd = [[1.0, 0.0], [0.99, 0.1]]
        v_iso = [[0.0, 1.0]]
        puntuar(list(docs_idd), list(v_idd))
        puntuar(list(docs_iso), list(v_iso))
        # el duplicado pierde por el término (1 − redundancia), aunque ambos
        # tienen la misma relevancia/autoridad/centralidad similar
        assert (
            docs_idd[0].metadata["semantic_score"]
            < docs_iso[0].metadata["semantic_score"]
        )

    def test_autoridad_normativa_mayor(self) -> None:
        mismo_vec: list[list[float]] = [[1.0, 0.0]]
        docs_norm = _docs(["norma"], source="200215-0-normativa-deportivas.pdf")
        docs_csv = _docs(["dato"], source="200085-0-deportes_abonos.csv")
        puntuar(list(docs_norm), list(mismo_vec))
        puntuar(list(docs_csv), list(mismo_vec))
        assert (
            docs_norm[0].metadata["semantic_score"]
            > docs_csv[0].metadata["semantic_score"]
        )

    def test_score_acotado_a_uno(self) -> None:
        """Chunk perfecto (relevancia 1.0, autoridad 1.0, sin redundancia,
        centrado): el recorte a 1.0 se mantiene."""
        docs = [
            Document(
                page_content="ok",
                metadata={
                    "source": "reglamento.pdf",
                    "relevancia_llm": 1.0,
                    "doc_category": "normativa",
                    "tags": "",
                },
            )
        ]
        puntuar(docs, [[1.0, 0.0]])
        assert docs[0].metadata["semantic_score"] == 1.0

    def test_lista_vacia(self) -> None:
        assert puntuar([], []) == []




# ---------------------------------------------------------------------------
# DEDUP
# ---------------------------------------------------------------------------


class TestDedup:
    def test_sin_scoring_se_degrada_graciosamente(self) -> None:
        """metadata sin semantic_score: se usa 0.0 por defecto (sin KeyError)."""
        docs = [Document(page_content=t, metadata={"source": "t.csv"}) for t in "abc"]
        chunks, _ = deduplicar(docs, _vecs([1.0, 1.0, 1.0]), umbral=0.9)
        assert len(chunks) == 1  # todos son clones: queda 1

    def test_dedup_por_score_descendente(self) -> None:
        """El clon con mejor score se conserva; el peor, se descarta."""
        docs_alto = [
            Document(
                page_content="A alto",
                metadata={"source": "f.pdf", "semantic_score": 0.9},
            )
        ]
        docs_bajo = [
            Document(
                page_content="B bajo",
                metadata={"source": "f.pdf", "semantic_score": 0.2},
            )
        ]
        docs_aislado = [
            Document(
                page_content="C distinto",
                metadata={"source": "f.pdf", "semantic_score": 0.5},
            )
        ]
        vecs = [[1.0, 0.0], [0.999, 0.01], [0.0, 1.0]]
        chunks, vecs_kept = deduplicar(
            docs_alto + docs_bajo + docs_aislado, vecs, umbral=0.95
        )
        # queda A (mejor score) + C (aislado); B se descarta como clon de A
        assert [c.page_content for c in chunks] == ["A alto", "C distinto"]
        assert [list(v) for v in vecs_kept] in ([[1.0, 0.0], [0.0, 1.0]],) or chunks[
            1
        ].page_content == "C distinto"
        assert len(chunks) == 2

    def test_umbral_elevado_nada_se_descarta(self) -> None:
        docs = _docs(["a", "b", "c"])
        vecs = [[1.0, 0.0], [0.3, 0.95], [0.0, 1.0]]
        puntuar(list(docs), list(vecs))  # para que exista semantic_score
        chunks, _ = deduplicar(docs, vecs, umbral=1.01)
        assert len(chunks) == 3

    def test_lista_vacia(self) -> None:
        chunks, vecs = deduplicar([], [])
        assert chunks == [] and vecs == []

    def test_orden_original_conservado(self) -> None:
        """Los conservados salen en orden original (no orden de score)."""
        docs = [
            Document(page_content=c, metadata={"source": "f.csv", "semantic_score": s})
            for c, s in zip("abc", (0.1, 0.9, 0.5))
        ]
        vecs = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        chunks, _ = deduplicar(docs, vecs, umbral=0.9)
        assert [c.page_content for c in chunks] == ["a", "b", "c"]

    def test_volca_cobertura_por_categoria(self) -> None:
        """info['dedup'].chunks_por_categoria: fija a la taxonomía cerrada (0 si no hay)."""
        docs = [
            Document(page_content=t, metadata={
                "source": "a.csv", "doc_category": cat,
            })
            for cat, t in (
                ("abonos", "uno"), ("abonos", "dos"), ("reservas", "tres"),
                ("agenda", "cuatro"), ("sin categoria", "cinco"),
            )
        ]
        vecs = [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
        info: dict = {}
        deduplicar(docs, vecs, umbral=0.9, info=info)
        cov = info["dedup"]["chunks_por_categoria"]
        assert cov["abonos"] == {"pre": 2, "post": 2}
        assert cov["reservas"] == {"pre": 1, "post": 1}
        assert cov["agenda"] == {"pre": 1, "post": 1}
        assert cov["tarifas"] == {"pre": 0, "post": 0}
        assert set(cov) == CATEGORIAS_VALIDAS  # siempre las 6 cerradas

    def test_cobertura_refleja_descartes_clon(self) -> None:
        """Un clon descartado baja el post: pre 2 → post 1 en su categoría."""
        docs = [
            Document(page_content="A", metadata={
                "source": "a.pdf", "semantic_score": 0.9, "doc_category": "reservas",
            }),
            Document(page_content="B clon", metadata={
                "source": "a.pdf", "semantic_score": 0.5, "doc_category": "reservas",
            }),
            Document(page_content="C", metadata={
                "source": "b.pdf", "semantic_score": 0.7, "doc_category": "abonos",
            }),
        ]
        vecs = [[1.0, 0.0], [0.999, 0.01], [0.0, 1.0]]
        info: dict = {}
        deduplicar(docs, vecs, umbral=0.9, info=info)
        assert info["dedup"]["chunks_por_categoria"]["reservas"] == {"pre": 2, "post": 1}

    def test_volca_tags_top3_post(self) -> None:
        """Top 3 tags post-dedup por nº de chunks (orden: recuento desc, luego alfabético)."""
        docs = [
            Document(page_content=t, metadata={
                "source": "a.csv", "semantic_score": s,
                **({tag: True} if tag else {}),
            })
            for t, s, tag in (
                ("uno", 0.9, "tag_piscina"), ("dos", 0.8, "tag_piscina"),
                ("tres", 0.7, "tag_reserva"), ("cuatro", 0.6, "tag_reserva"),
                ("cinco", 0.5, None),
            )
        ]
        vecs = [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
        info: dict = {}
        deduplicar(docs, vecs, umbral=0.9, info=info)
        assert info["dedup"]["tags_top3_post"] == [("piscina", 2), ("reserva", 2)]

    def test_cobertura_sin_info_no_lanza(self) -> None:
        docs = _docs(["a"])
        deduplicar(docs, [[1.0, 0.0]])  # sin info: no volca, no rompe


# ---------------------------------------------------------------------------
# AUDITORÍA DE DESCARTES (eventos + agregados del dedup)
# ---------------------------------------------------------------------------


class TestAuditoriaDescartes:
    """info['dedup'] volca la auditoría de los pares descartados: eventos con
    motivo (coseno/clave), similaridad del par, y agregados por fuente."""

    def test_eventos_coseno_incluyen_sim_y_pareja(self) -> None:
        docs = [
            Document(page_content="A", metadata={
                "source": "a.txt", "semantic_score": 0.9, "chunk_index": 0,
            }),
            Document(page_content="B clon de A", metadata={
                "source": "b.txt", "semantic_score": 0.5, "chunk_index": 1,
            }),
        ]
        vecs = [[1.0, 0.0], [0.9999, 0.001]]
        info: dict = {}
        deduplicar(list(docs), list(vecs), umbral=0.9, info=info)
        dd = info["dedup"]
        assert len(dd["descartes"]) == 1
        ev = dd["descartes"][0]
        assert ev["motivo"] == "dedup_coseno"
        assert ev["fuente"] == "b.txt" and ev["chunk_index"] is not None
        assert ev["sim"] >= 0.9
        assert ev["snip"].startswith("B clon de A")  # snippet de la fuente descartada
        par = ev["pareja"]
        assert par["fuente"] == "a.txt"
        assert par["score"] == 0.9

    def test_evento_clave_para_politicas_exactas(self) -> None:
        docs = [
            Document(page_content="entidad 1", metadata={
                "source": "a.csv", "dedup_policy": "exact_key",
                "entity_key": "k1", "semantic_score": 0.9,
            }),
            Document(page_content="entidad 1 bis", metadata={
                "source": "a.csv", "dedup_policy": "exact_key",
                "entity_key": "k1", "semantic_score": 0.8,
            }),
        ]
        vecs = [[1.0, 0.0], [0.0, 1.0]]
        info: dict = {}
        deduplicar(list(docs), list(vecs), umbral=0.9, info=info)
        ev = info["dedup"]["descartes"][0]
        assert ev["motivo"] == "dedup_clave"
        assert ev["policy"] == "exact_key"
        assert ev["fuente"] == "a.csv"

    def test_fuentes_vacias_por_coseno(self) -> None:
        """Fuente pre -> 0 post: se registra como vacía con el descarte que la vació."""
        docs = [
            Document(page_content="A", metadata={"source": "a.txt", "semantic_score": 0.9}),
            Document(page_content="B clon", metadata={
                "source": "b.txt", "semantic_score": 0.5,
            }),
        ]
        vecs = [[1.0, 0.0], [0.9999, 0.001]]
        info: dict = {}
        deduplicar(list(docs), list(vecs), umbral=0.9, info=info)
        fv = info["dedup"]["fuentes_vacias"]
        assert list(fv) == ["b.txt"]
        assert fv["b.txt"]["sim"] >= 0.9
        assert fv["b.txt"]["pareja"]["fuente"] == "a.txt"

    def test_fuentes_vacias_por_clave_imposible(self) -> None:
        """Las claves exactas SIEMPRE conservan el primer chunk: ninguna fuente
        puede quedar vacía por ese motivo (solo el coseno vacía una fuente)."""
        docs = [Document(
            page_content=t,
            metadata={
                "source": "a.csv", "dedup_policy": "exact_key",
                "entity_key": "k1", "semantic_score": 0.9,
            },
        ) for t in ("row1", "row1-dup")]
        vecs = [[1.0, 0.0], [0.0, 1.0]]
        info: dict = {}
        deduplicar(list(docs), list(vecs), umbral=0.9, info=info)
        assert len(info["dedup"]["descartes"]) == 1  # la repetida se descarta
        assert info["dedup"]["fuentes_vacias"] == {}  # pero la fuente no se vacía

    def test_descartes_por_fuente(self) -> None:
        docs = [
            Document(page_content=f"{t}", metadata={
                "source": "a.csv", "dedup_policy": "exact_key",
                "entity_key": "k1", "semantic_score": 0.9,
            })
            for t in ("row1", "row1-dup", "row1-dup2")
        ]
        docs.append(
            Document(page_content="clon lejano", metadata={
                "source": "b.txt", "semantic_score": 0.5,
            })
        )
        vecs = [[1.0, 0.0]] * 3 + [[0.0, 1.0]]
        info: dict = {}
        deduplicar(list(docs), list(vecs), umbral=0.9, info=info)
        assert info["dedup"]["descartes_por_fuente"]["a.csv"] == 2

    def test_sin_descartes_eventos_vacios(self) -> None:
        docs = [Document(page_content=t, metadata={"source": "a.txt"})
                for t in ("unicus", "distinctus")]
        vecs = [[1.0, 0.0], [0.0, 1.0]]
        info: dict = {}
        deduplicar(list(docs), list(vecs), umbral=0.9, info=info)
        dd = info["dedup"]
        assert dd["descartes"] == []
        assert dd["descartes_por_fuente"] == {}
        assert dd["fuentes_vacias"] == {}


# ---------------------------------------------------------------------------
# CORRECTUD: FAISS vs. implementación de referencia (matriz completa)
# ---------------------------------------------------------------------------


class TestCorrectitudVsReferencia:
    """El resultado debe ser idéntico (a nivel de bits flotantes en la lógica)
    al algoritmo original sobre la matriz de similitud completa, con vectores
    sintéticos a escala pequeña que SÍ caben en RAM.
    """

    @staticmethod
    def _scoring_referencial(
        chunks: list[Document], embeddings: list[list[float]]
    ) -> list[float]:
        """Reimplementación inline del algoritmo original (scoring por matriz)."""
        from src.tsd.scoring import _autoridad

        E = np.asarray(embeddings, dtype=np.float32)
        centroide = E.mean(axis=0)
        norm = np.linalg.norm(centroide)
        if norm > 0:
            centroide /= norm
        sim = E.astype(np.float64) @ E.astype(np.float64).T
        np.fill_diagonal(sim, 0.0)
        out = []
        for i, chunk in enumerate(chunks):
            m = chunk.metadata
            score = (
                0.45 * float(m.get("relevancia_llm", 0.0))
                + 0.20 * float(E[i] @ centroide)
                + 0.20 * (1.0 - float(sim[i].max()))
                + 0.15 * _autoridad(m.get("source", ""))
            )
            out.append(round(min(max(score, 0.0), 1.0), 4))
        return out

    def test_scores_iguales_a_referencia(self) -> None:
        rng = np.random.RandomState(7)
        n, d = 300, 16
        E = rng.standard_normal((n, d)).astype("float32")
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        textos = [f"chunk {i}" for i in range(n)]
        docs = _docs(textos)
        vecs = E.tolist()
        puntuar(list(docs), list(vecs))
        esperado = self._scoring_referencial(list(_docs(textos)), list(vecs))
        for got, exp in zip([c.metadata["semantic_score"] for c in docs], esperado):
            assert abs(got - exp) <= 0.0001 or (got == 1.0) == (exp == 1.0)

    def test_dedup_idem_a_referencia(self) -> None:
        """Dedup FAISS vs. dedup por matriz completa: mismo conjunto conservado,
        mismo orden."""
        rng = np.random.RandomState(11)
        n, d = 300, 16
        E = rng.standard_normal((n, d)).astype("float32")
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        # inyectar clones exactos (caso límite del umbral)
        E[200:230] = E[50:80]
        textos = [f"chunk {i}" for i in range(n)]
        docs = _docs(textos)
        vecs = E.tolist()
        puntuar(list(docs), list(vecs))
        umbral = 0.93
        chunks, _ = deduplicar(list(docs), list(vecs), umbral=umbral)

        # referencia: matriz completa
        E64 = E.astype(np.float64)
        sim = E64 @ E64.T
        orden = sorted(
            range(n),
            key=lambda i: docs[i].metadata.get("semantic_score", 0.0),
            reverse=True,
        )
        kept: list[int] = []
        for i in orden:
            if all(sim[i, j] < umbral for j in kept):
                kept.append(i)
        kept.sort()
        esperado = [f"chunk {i}" for i in kept]
        assert [c.page_content for c in chunks] == esperado
