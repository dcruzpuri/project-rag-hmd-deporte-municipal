"""
tests/test_informe_e2e.py
Prueba de humo extremo a extremo (offline, proveedor fake): ejecutar_pipeline()
genera el dict de métricas con el que se renderiza el informe de indexación,
incluyendo las nuevas métricas de runtime (fases, scoring, dedup, índice).

Ejecutar:  pytest tests/test_informe_e2e.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def _etiquetar_fake(documentos):
    for doc in documentos:
        doc.metadata.update({"doc_category": "tarifas", "tags": "abono",
                             "relevancia_llm": 0.8})
    return documentos


def test_pipeline_genera_metricas_para_el_informe(
    tmp_path: Path, monkeypatch
) -> None:
    import config
    from src import embed as embed_mod
    from src import pipeline as pipeline_mod
    from src.index import obtener_cliente_chroma
    from src.informe import generar_informe

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text(
        "Precio del abono de piscina: 10 euros.", encoding="utf-8")
    (corpus / "b.txt").write_text(
        "Instalación deportiva municipal del barrio.", encoding="utf-8")

    # proveedor fake determinista (4 dims, sin red ni Ollama)
    monkeypatch.setitem(embed_mod._PROVIDERS, "ollama",
                        lambda textos, model, dim: np.full((len(textos), 4), 0.5, dtype=np.float32).tolist())
    # sella el entorno de ejecución: si el .env apunta a otro proveedor/dim u
    # a TSD apagado, la prueba sería del entorno y no del pipeline
    monkeypatch.setattr(embed_mod, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(embed_mod, "EMBED_DIM", 4)
    monkeypatch.setattr(config, "EMBED_DIM", 4)
    monkeypatch.setattr(pipeline_mod, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(pipeline_mod, "TAG_SCORING_DEDUP", True)
    monkeypatch.setattr(pipeline_mod, "EXPORT_EMBEDDINGS", False)
    monkeypatch.setattr(
        pipeline_mod, "verificar_modelo_disponible",
        lambda: {"disponible": True, "dim_modelo": 4,
                 "verificado_por": "online", "aviso": None},
    )
    monkeypatch.setattr(pipeline_mod, "etiquetar", _etiquetar_fake)

    # el pipeline escribe el informe por defecto en ./output: aislamos ese side effect
    monkeypatch.chdir(tmp_path)

    report = pipeline_mod.ejecutar_pipeline(
        str(corpus),
        persist_dir=str(tmp_path / "chroma"),
        collection_name="e2e_test",
        recreate_index=True,
    )

    # métricas de la ejecución presentes en el dict del informe
    assert report["num_documentos"] == 2
    assert report["num_chunks_pre_dedup"] > 0
    assert report["num_chunks_post_dedup"] <= report["num_chunks_pre_dedup"]
    assert report["dim_embedding"] == 4
    assert report["chunk_stats"]["min"] >= 0
    assert set(report["fases"]) >= {"LOAD", "CHUNK", "EMBED", "INDEX"}
    assert report["scoring"]["semantic_score_media"] >= 0
    assert report["dedup"]["chunks_pre"] == report["num_chunks_pre_dedup"]
    assert report["indice"]["vectores_insertados"] == report["num_chunks_post_dedup"]
    assert report["preflight"]["verificado_por"] == "online"

    # el informe referencia la carpeta GUID creada por ChromaDB
    guid_chroma = report["indice"]["guid_chroma"]
    assert guid_chroma
    informes = list((tmp_path / "output").glob(f"informe_index_{guid_chroma[:8]}_*.md"))
    assert len(informes) == 1
    ruta = informes[0]
    texto = ruta.read_text(encoding="utf-8")
    assert texto.startswith("# Informe de indexación")
    for seccion in ("## 1. Parámetros aplicados", "## 2. Preflight",
                    "## 3. Métricas por fase", "## 4. Chunking",
                    "## 5. Índice final (ChromaDB)", "## 6. TSD (scoring + dedup)",
                    "## 7. Señales y criterios de decisión"):
        assert seccion in texto

    # el índice quedó en la colección pedida
    cliente = obtener_cliente_chroma(str(tmp_path / "chroma"))
    assert cliente.get_collection("e2e_test").count() == report["num_chunks_post_dedup"]

    # generar_informe también acepta dict parcial (mínimo)
    ruta_min = generar_informe(
        {"rutas": ["data"], "parametros": []},
        tmp_path / "informe_min.md")
    assert ruta_min.read_text(encoding="utf-8").startswith("# Informe de indexación")
