"""
tests/test_informe.py
Tests unitarios offline para src/informe.py (generación del informe de indexación)
y las métricas de runtime que vuelcan scoring/dedup vía el parámetro ``info``.

Ejecutar:  pytest tests/test_informe.py -v
"""
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from src.informe import generar_informe
from src.tsd.dedup import deduplicar
from src.tsd.scoring import puntuar


def _datos(**overrides) -> dict[str, Any]:
    """Dict de métricas del pipeline con valores de prueba mínimos."""
    base: dict[str, Any] = {
        "rutas": ["data"],
        "parametros": [
            ("EMBED_PROVIDER", "ollama", "proveedor de embeddings"),
            ("EMBED_MODEL", "qwen3-embedding:4b", "modelo de embeddings"),
            ("EMBED_DIM_DECLARADO", 3072, "tope declarado en .env"),
            ("EMBED_DIM_MODELO", 2560, "dimensión máxima del modelo (EMBED_DIM_MAX_*)"),
            ("EMBED_DIM", 2560, "dimensión efectiva del índice (min(dim modelo, declarado))"),
        ],
        "preflight": {
            "verificado_por": "online",
            "dim_modelo": 2560,
            "aviso": None,
        },
        "fases": {"LOAD": 0.1, "CLEAN": 0.0, "CHUNK": 0.0, "EMBED": 1.0, "INDEX": 0.5},
        "resumen_fases": {
            "LOAD": "2 documentos cargados",
            "CLEAN": "2 documentos normalizados",
            "CHUNK": "3 chunks generados (min 10, media 20)",
            "EMBED": "3 vectores de 8 dims",
            "INDEX": "3 vectores insertados de 3 totales",
        },
        "num_documentos_cargados": 2,
        "num_documentos_normalizados": 2,
        "num_chunks_pre_dedup": 3,
        "num_chunks_post_dedup": 3,
        "chunks_descartados": 0,
        "dim_embedding": 8,
        "dim_msg": "",
        "chunk_stats": {"min": 10, "p25": 12, "media": 20, "p50": 20, "p75": 30,
                        "max": 40, "cortos": 0, "sin_trocear": 0},
        "fuentes": [
            ("a.txt", {"docs": 1, "chunks_pre": 2, "chunks_post": 1}),
            ("b.csv", {"docs": 1, "chunks_pre": 1, "chunks_post": 1}),
        ],
        "integridad": {"n": 3, "dim": 8, "dim_ok": 3, "nan_inf": 0, "cero": 0,
                       "norma_min": 1.0, "norma_media": 1.0, "norma_max": 1.0},
        "scoring": None,
        "dedup": None,
        "indice": {
            "nombre": "deporte_municipal",
            "space": "cosine",
            "dim": 8,
            "vectores_insertados": 3,
            "vectores_totales": 3,
            "recreado": True,
        },
        "tiempo_total_s": 2.0,
    }
    base.update(overrides)
    return base


class TestGenerarInforme:
    def test_escriven_el_md_en_output(self, tmp_path: Path) -> None:
        ruta = generar_informe(_datos(), tmp_path / "informe.md")
        assert ruta.exists()
        texto = ruta.read_text(encoding="utf-8")
        assert texto.startswith("# Informe de indexación")
        assert "`EMBED_PROVIDER`" in texto

    def test_tabla_parametros_incluye_variable_y_valor(self, tmp_path: Path) -> None:
        ruta = generar_informe(_datos(), tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        assert "qwen3-embedding:4b" in texto
        assert "| 2560 |" in texto

    def test_metricas_de_fase_y_chunking(self, tmp_path: Path) -> None:
        ruta = generar_informe(_datos(), tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        assert "2 documentos cargados" in texto
        assert "| min | 10 |" in texto
        assert "| max | 40 |" in texto
        assert "deporte_municipal" in texto
        assert "3 vectores" in texto

    @staticmethod
    def _dedup_cobertura(
        cobertura: dict, tags_top3: list[tuple[str, int]] | None = None
    ) -> dict[str, Any]:
        base = {
            "umbral": 0.9, "chunks_pre": 10, "chunks_post": 2,
            "descartados_exactos": 1, "descartados_semantico": 7,
            "descartados_total": 8, "descartados_pct": 80.0, "tiempo_s": 1.932,
            "chunks_por_categoria": cobertura,
        }
        if tags_top3 is not None:
            base["tags_top3_post"] = tags_top3
        return base

    def test_seccion_63_cobertura_por_categoria(self, tmp_path: Path) -> None:
        """La 6.3 lista las 6 categorías fijas (pre/post) y la fila top-3 de tags."""
        dedup = self._dedup_cobertura({
            "tarifas": {"pre": 3, "post": 2},
            "normativa": {"pre": 4, "post": 4},
            "reservas": {"pre": 1, "post": 0},
            "abonos": {"pre": 2, "post": 2},
            "instalaciones": {"pre": 2, "post": 1},
            "agenda": {"pre": 0, "post": 0},
        }, tags_top3=[("piscina", 5), ("reserva", 3), ("precio", 1)])
        ruta = generar_informe(_datos(dedup=dedup), tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        assert "### 6.3 Cobertura por categoría" in texto
        assert "| agenda | 0 | 0 |" in texto
        assert "| reservas | 1 | 0 |" in texto
        assert "piscina (5)" in texto and "reserva (3)" in texto and "precio (1)" in texto
        # la categoría vacía (post 0) dispara la señal de decisión
        assert "la categoría reservas" in texto and "vacía" in texto.lower()

    def test_seccion_63_sin_cobertura_no_renderiza(self, tmp_path: Path) -> None:
        """Sin TSD (o sin cobertura volcada) no hay 6.3 ni señal de categoría vacía."""
        ruta = generar_informe(_datos(), tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        assert "### 6.3" not in texto
        assert "Bloque TSD desactivado" in texto

    def test_cobertura_completa_sin_senal_vacia(self, tmp_path: Path) -> None:
        dedup = self._dedup_cobertura({
            **{cat: {"pre": 2, "post": 2} for cat in (
                "tarifas", "normativa", "reservas", "abonos",
                "instalaciones", "agenda")},
        })
        ruta = generar_informe(_datos(dedup=dedup), tmp_path / "informe.md")
        assert "**Cobertura:**" not in ruta.read_text(encoding="utf-8")

    def test_senales_con_dedup_activa(self, tmp_path: Path) -> None:
        dedup = {
            "umbral": 0.9, "chunks_pre": 10, "chunks_post": 2,
            "descartados_exactos": 1, "descartados_semantico": 7,
            "descartados_total": 8, "descartados_pct": 80.0, "tiempo_s": 1.932,
        }
        scoring = {
            "semantic_score_min": 0.3, "semantic_score_media": 0.6,
            "semantic_score_max": 0.9, "centralidad_media": 0.5,
            "redundancia_media": 0.2, "score_buenos_pct": 60.0,
            "score_buenos_n": 6, "chunks_n": 10, "tiempo_s": 0.5,
        }
        ruta = generar_informe(_datos(dedup=dedup, scoring=scoring),
                               tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        assert "0.9" in texto
        assert "80.0" in texto
        assert "descartado" in texto.lower()
        # ≥ 0.6: recuento absoluto N de total + tiempo en ms (sin falsos 0.0 s)
        assert "| chunks con score ≥ 0.6 | 6 de 10 (60.0 %) |" in texto
        assert "| tiempo | 500.0 ms |" in texto
        assert "| 1.93 s |" in texto

    def test_senales_dedup_activa_sin_info(self, tmp_path: Path) -> None:
        ruta = generar_informe(_datos(), tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        assert "Bloque TSD desactivado" in texto

    def test_puerto_por_default_output(self, tmp_path: Path) -> None:
        (tmp_path / "output").mkdir()
        ruta = generar_informe(_datos(), tmp_path / "output" / "informe_indexacion.md")
        assert ruta == tmp_path / "output" / "informe_indexacion.md"
        assert ruta.exists()

    def test_nombre_por_defecto_referencia_guid_de_chroma(self, tmp_path: Path,
                                                          monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        datos = _datos()
        datos["indice"]["guid_chroma"] = "3bdb6429-68dd-483f-91f7-27d3bc8b7e48"
        ruta = generar_informe(datos)
        assert ruta.name.startswith(
            "informe_index_3bdb6429_"
        )

    def test_senal_chunks_cortos(self, tmp_path: Path) -> None:
        stats = {"min": 5, "p25": 10, "media": 100, "p50": 100, "p75": 200,
                 "max": 900, "cortos": 42}
        ruta = generar_informe(_datos(chunk_stats=stats), tmp_path / "informe.md")
        assert "42" in ruta.read_text(encoding="utf-8")

    def test_senal_chunk_corto_singular(self, tmp_path: Path) -> None:
        """Un solo corto: '1 chunk', no '1 chunks'."""
        stats = {"min": 1, "p25": 5, "media": 100, "p50": 100, "p75": 200,
                 "max": 900, "cortos": 1}
        ruta = generar_informe(_datos(chunk_stats=stats), tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        assert "hay 1 chunk < 50 caracteres" in texto
        assert "1 chunks" not in texto

    def test_fuentes_vacia_post_dedup_alerta_critica(self, tmp_path: Path) -> None:
        """Fuente con chunks pre y 0 post: alerta crítica en sección 4 y señal 7."""
        ruta = generar_informe(
            _datos(fuentes=[("a.pdf", {"docs": 1, "chunks_pre": 1, "chunks_post": 0})]),
            tmp_path / "informe.md",
        )
        texto = ruta.read_text(encoding="utf-8")
        assert "| a.pdf | 1 | 1 | 0 |" in texto
        assert "Fuente vacía post-dedup (crítica)" in texto  # nota bloqueada en la tabla
        assert "**Fuente vacía (crítica):** `a.pdf`" in texto  # señal explícita
        assert "DEDUP_UMBRAL" in texto

    def test_fuentes_sin_vacias_sin_alerta(self, tmp_path: Path) -> None:
        """Sin fuentes vacías no aparece la alerta crítica (base de _datos)."""
        ruta = generar_informe(_datos(), tmp_path / "informe.md")
        assert "Fuente vacía" not in ruta.read_text(encoding="utf-8")

    def test_dim_modelo_no_declarada(self, tmp_path: Path) -> None:
        """EMBED_DIM_MAX_* vacía: la fila MODELO dice 'no declarada' y la
        sección 2 lo deja constancia (el preflight online no expone la dim)."""
        datos = _datos(
            parametros=[
                ("EMBED_PROVIDER", "ollama", "proveedor de embeddings"),
                ("EMBED_DIM_MODELO", "no declarada",
                 ("EMBED_DIM_MAX_* no declarada: el preflight solo verificó la "
                  "disponibilidad (no la dimensión nativa del modelo)")),
                ("EMBED_DIM", 2560, "dimensión efectiva del índice"),
            ],
            preflight={"verificado_por": "online", "dim_modelo": None,
                       "aviso": None},
        )
        ruta = generar_informe(datos, tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        assert "`EMBED_DIM_MODELO` | no declarada |" in texto
        assert "`EMBED_DIM_MAX_*`):** no declarada" in texto

    def test_seccion4_fuentes_integridad_y_tres_dims(self, tmp_path: Path) -> None:
        """Sección 4: tabla fuentes→(docs, chunks pre/post) y tres filas EMBED_DIM;
        sección 5: línea de integridad de los vectores."""
        ruta = generar_informe(_datos(), tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        assert "## 4. Corpus y chunking (longitudes en caracteres)" in texto
        assert "| a.txt | 1 | 2 | 1 |" in texto
        assert "| **total** | 2 | 3 | 2 |" in texto
        assert "EMBED_DIM_DECLARADO" in texto and "EMBED_DIM_MODELO" in texto
        assert "| 3072 |" in texto
        assert "- **Integridad:** 3 vectores · dim 8 · NaN/Inf 0 · ceros 0" in texto
        assert "| íntegras (`no_chunk`) | 0 |" in texto

    def test_orden_secciones_seis(self, tmp_path: Path) -> None:
        """Orden único 6.1 < 6.2 < 6.3, sin 'útiles', umbral interno documentado."""
        dedup = self._dedup_cobertura({
            "tarifas": {"pre": 3, "post": 2},
            "normativa": {"pre": 2, "post": 2},
            "reservas": {"pre": 1, "post": 1},
            "abonos": {"pre": 2, "post": 2},
            "instalaciones": {"pre": 1, "post": 1},
            "agenda": {"pre": 1, "post": 1},
        })
        scoring = {
            "semantic_score_min": 0.3, "semantic_score_media": 0.6,
            "semantic_score_max": 0.9, "centralidad_media": 0.5,
            "redundancia_media": 0.2, "score_buenos_pct": 60.0,
            "score_buenos_n": 6, "chunks_n": 10, "tiempo_s": 0.5,
        }
        ruta = generar_informe(_datos(dedup=dedup, scoring=scoring), tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        i61, i62, i63 = (texto.index(x) for x in ("### 6.1", "### 6.2", "### 6.3"))
        assert i61 < i62 < i63
        assert "útiles" not in texto
        assert "NO es precisión ni relevancia" in texto
        assert "| umbral coseno (`DEDUP_UMBRAL`) | 0.9 |" in texto

    def test_seccion_61_redundancia_por_politica(self, tmp_path: Path) -> None:
        """La 6.1 muestra la tabla de redundancia por política (umbral desde el volcado)."""
        scoring = {
            "semantic_score_min": 0.3, "semantic_score_media": 0.6,
            "semantic_score_max": 0.9, "centralidad_media": 0.5,
            "redundancia_media": 0.2, "score_buenos_pct": 60.0,
            "score_buenos_n": 6, "chunks_n": 10, "tiempo_s": 0.5,
            "redundancia_umbral": 0.93,
            "redundancia_por_politica": {
                "exact_key": {"n": 854, "media": 0.8215, "p50": 0.7982,
                              "p90": 0.989, "pct_sup_umbral": 17.1},
                "group_only": {"n": 13186, "media": 0.984, "p50": 0.988,
                               "p90": 0.9945, "pct_sup_umbral": 99.8},
                "semantica": {"n": 739, "media": 0.8664, "p50": 0.9199,
                              "p90": 0.9286, "pct_sup_umbral": 0.0},
            },
        }
        ruta = generar_informe(_datos(scoring=scoring), tmp_path / "informe.md")
        texto = ruta.read_text(encoding="utf-8")
        assert "| política | n | media | p50 | p90 | % ≥ umbral |" in texto
        assert "| exact_key | 854 | 0.8215 |" in texto
        assert "| 0.0 % |" in texto and "| 99.8 % |" in texto
        assert "estructural" in texto

    def test_seccion61_sin_volcado_no_tabla_politica(self, tmp_path: Path) -> None:
        """Sin la métrica volcada por el scoring no se muestra la tabla por política."""
        scoring = {
            "semantic_score_min": 0.3, "semantic_score_media": 0.6,
            "semantic_score_max": 0.9, "centralidad_media": 0.5,
            "redundancia_media": 0.2, "score_buenos_pct": 60.0,
            "score_buenos_n": 6, "chunks_n": 10, "tiempo_s": 0.5,
        }
        ruta = generar_informe(_datos(scoring=scoring), tmp_path / "informe.md")
        assert "| política | n |" not in ruta.read_text(encoding="utf-8")


class TestMetricasRuntime:
    """scoring/dedup vuelcan sus métricas en el dict ``info`` del pipeline."""

    def test_scoring_volca_metricas(self) -> None:
        docs = [Document(page_content=t,
                         metadata={"source": "a.pdf", "relevancia_llm": 0.5})
                for t in ("hola mundo", "otro texto largo", "tercero diferente")]
        vecs = [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
        info: dict[str, Any] = {}
        puntuar(list(docs), list(vecs), info=info)
        sc = info["scoring"]
        assert 0.0 <= sc["semantic_score_min"] <= sc["semantic_score_max"]
        assert sc["score_buenos_pct"] == float(sc["score_buenos_pct"])
        assert sc["tiempo_s"] >= 0

    def test_scoring_volca_redundancia_por_politica(self) -> None:
        import config

        docs = [
            Document(page_content="fila", metadata={"source": "a.csv",
                                                     "dedup_policy": "exact_key",
                                                     "relevancia_llm": 0.5}),
            Document(page_content="grupo", metadata={"source": "b.csv",
                                                      "dedup_policy": "group_only",
                                                      "relevancia_llm": 0.5}),
            Document(page_content="texto cualquiera", metadata={
                "source": "c.txt", "relevancia_llm": 0.5}),
        ]
        vecs = [[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]]
        info: dict[str, Any] = {}
        puntuar(list(docs), list(vecs), info=info)
        sc = info["scoring"]
        assert set(sc["redundancia_por_politica"]) == {"exact_key", "group_only", "semantica"}
        assert sc["redundancia_por_politica"]["semantica"]["n"] == 1
        assert sc["redundancia_umbral"] == config.DEDUP_UMBRAL

    def test_dedup_volca_metricas(self) -> None:
        docs = [Document(page_content=t,
                         metadata={"source": "a.csv", "semantic_score": s})
                for t, s in (("uno", 0.9), ("dos casi como uno", 0.8),
                             ("algo totalmente distinto", 0.3))]
        vecs = [[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]]
        info: dict[str, Any] = {}
        deduplicar(list(docs), list(vecs), umbral=0.9, info=info)
        dd = info["dedup"]
        assert dd["umbral"] == 0.9
        assert dd["chunks_pre"] == 3
        assert dd["chunks_post"] == dd["chunks_pre"] - dd["descartados_total"]
        assert 0.0 <= dd["descartados_pct"] <= 100.0
        assert dd["tiempo_s"] >= 0

    def test_sin_info_no_lanza(self) -> None:
        docs = [Document(page_content="solo uno", metadata={"source": "x.csv"})]
        vecs = [[1.0, 0.0]]
        assert puntuar(list(docs), list(vecs)) is not None
        assert deduplicar(list(docs), list(vecs)) is not None
