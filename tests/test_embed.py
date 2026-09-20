"""
tests/test_embed.py
Tests offline (sin red, sin Ollama) para el preflight de disponibilidad del
modelo de embeddings y para la seguridad de dimensiones del pipeline.

Ejecutar:  pytest tests/test_embed.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any

import numpy as np
import requests

from src import embed as embed_mod
from src.embed import (
    _corte_dim,
    _dim_max_cache,
    _normalizar,
    embeddear,
    verificar_modelo_disponible,
)
from src.pipeline import _comprobar_dim

# =============== Preflight: Ollama =================

class _RespFake:
    def __init__(self, data: dict[str, Any], status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def _preflight_ollama(monkeypatch, nombres: list[str] | None = None):
    """Sustituye GET {base}/api/tags por una respuesta fija y devuelve el preflight.
    Si ``nombres`` es None no hay servidor (ConnectionError)."""
    def _get(url, timeout=None):
        assert url == "http://127.0.0.1:11434/api/tags"
        if nombres is None:
            raise requests.ConnectionError("connection refused")
        return _RespFake({"models": [{"name": n} for n in nombres]})
    monkeypatch.setattr(embed_mod.requests, "get", _get)
    return verificar_modelo_disponible({
        "EMBED_PROVIDER": "ollama",
        "EMBED_MODEL": "qwen3-embedding:4b",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
    })


class TestPreflightOllama:
    """El preflight consulta GET {OLLAMA_BASE_URL}/api/tags (servidor local)."""

    @pytest.mark.parametrize("nombres,disponible", [
        (["qwen3-embedding:4b"], True),           # exacto (namespace:tag)
        (["qwen3-embedding:4b-instruct"], False), # otro tag de la misma base
        (["llama3.2", "mixtral"], False),          # el modelo no está en el servidor
        ([], False),                               # servidor vacío
    ])
    def test_online(self, monkeypatch, nombres, disponible):
        if disponible:
            res = _preflight_ollama(monkeypatch, nombres)
            assert res["verificado_por"] == "online"
            assert res["disponible"] is True
        else:
            with pytest.raises(RuntimeError, match="NO está disponible"):
                _preflight_ollama(monkeypatch, nombres)

    def test_sin_servidor_cae_al_cache_env(self, monkeypatch):
        monkeypatch.setenv("EMBED_DIM_MAX_OLLAMA", "2560")
        res = _preflight_ollama(monkeypatch, None)  # ConnectionError
        assert res["verificado_por"] == "cache_env"
        assert res["dim_modelo"] == 2560
        assert res["disponible"] is True
        assert res["aviso"] is not None

    def test_5xx_cae_al_cache(self, monkeypatch):
        def _raise(url, timeout=None):
            raise requests.HTTPError("500", response=_RespFake({}, status_code=500))

        monkeypatch.setattr(embed_mod.requests, "get", _raise)
        monkeypatch.setenv("EMBED_DIM_MAX_OLLAMA", "2560")
        res = verificar_modelo_disponible({
            "EMBED_PROVIDER": "ollama",
            "EMBED_MODEL": "qwen3-embedding:4b",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        })
        assert res["verificado_por"] == "cache_env"


# =============== Preflight: HuggingFace ============

class TestPreflightHuggingFace:
    def test_repo_existe_online(self, monkeypatch):
        monkeypatch.setattr(embed_mod, "repo_exists", lambda m, token=None: True)
        res = verificar_modelo_disponible({
            "EMBED_PROVIDER": "huggingface",
            "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        })
        assert res["verificado_por"] == "online"
        assert res["disponible"] is True

    def test_repo_no_existe_online(self, monkeypatch):
        monkeypatch.setattr(embed_mod, "repo_exists", lambda m, token=None: False)
        with pytest.raises(RuntimeError, match="NO está disponible"):
            verificar_modelo_disponible({
                "EMBED_PROVIDER": "huggingface",
                "EMBED_MODEL": "usuario/modelo-que-no-existe-xyz",
            })

    def test_red_cortada_cae_al_cache(self, monkeypatch):
        def _raise(m, token=None):
            raise OSError("ConnectionError")

        monkeypatch.setattr(embed_mod, "repo_exists", _raise)
        monkeypatch.setenv("EMBED_DIM_MAX_HF", "1024")
        res = verificar_modelo_disponible({
            "EMBED_PROVIDER": "huggingface",
            "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        })
        assert res["verificado_por"] == "cache_env"
        assert res["dim_modelo"] == 1024


# =============== Preflight: Google ================

class TestPreflightGoogle:
    def test_sin_api_key_cae_al_cache(self, monkeypatch):
        # Aíslandolo: el .env real puede llevar una GOOGLE_API_KEY, y entonces
        # la comprobación iría a online. Pateamos la constante a vacía (sin key).
        monkeypatch.setattr(embed_mod, "GOOGLE_API_KEY", "")
        monkeypatch.setenv("EMBED_DIM_MAX_GOOGLE", "768")
        res = verificar_modelo_disponible({
            "EMBED_PROVIDER": "google",
            "EMBED_MODEL": "gemini-embedding-2",
            "GOOGLE_API_KEY": "",
        })
        assert res["verificado_por"] == "cache_env"
        assert res["dim_modelo"] == 768


# =============== Caché offline EMBED_DIM_MAX_* ============

class TestDimMaxCache:
    def test_cache_vacio_devuelve_none(self):
        assert _dim_max_cache("ollama", {}) is None

    def test_cache_declarado(self, monkeypatch):
        monkeypatch.setenv("EMBED_DIM_MAX_HF", "1536")
        assert _dim_max_cache("huggingface", {}) == 1536

    def test_meta_por_sobre_env(self, monkeypatch):
        monkeypatch.setenv("EMBED_DIM_MAX_OLLAMA", "2560")
        assert _dim_max_cache("ollama", {"EMBED_DIM_MAX_OLLAMA": "1024"}) == 1024


# =============== Recorte de dimensión (embed.py) ============

class TestCorteDim:
    """_corte_dim(): cap del tamaño de los vectores (prefijo + renormalizar)."""

    def test_dim_cap_none_no_op(self):
        vectores = [[0.3, 0.4, 0.9], [0.0, 0.0, 0.5]]
        assert _corte_dim(vectores, None) is vectores

    def test_dim_cap_mayor_no_op(self):
        vectores = [[0.3, 0.4], [0.0, 0.5]]
        assert _corte_dim(vectores, 5) is vectores

    def test_recorta_prefijo_y_renormaliza(self):
        # vectores unitarios de 5 dims; el cap a 3 debe conservar la dirección
        vectores = _normalizar([[3.0, 4.0, 1.0, 1.0, 0.5]])
        resultado = _corte_dim(vectores, 3)
        assert len(resultado) == 1
        assert len(resultado[0]) == 3
        # misma dirección que (3, 4, 1), renormalizada: el recorte solo escala
        esperado = (np.array([3.0, 4.0, 1.0])
                    / np.linalg.norm([3.0, 4.0, 1.0]))
        assert np.allclose(resultado[0], esperado, atol=1e-5)

    def test_recorte_de_prefijo_cero_no_explora(self):
        # prefijo todo cero → el guard de norma 0 evita la división por cero
        vectores = _normalizar([[0.0, 0.0, 0.0, 1.0, 0.0]])
        resultado = _corte_dim(vectores, 3)
        assert resultado[0] == [0.0, 0.0, 0.0]


class TestEmbeddearCap:
    """embeddear(): el cap por defecto es EMBED_DIM (backstep cliente)."""

    def test_cap_por_defecto_embed_dim(self, monkeypatch):
        # el proveedor devuelve 2560 dims; EMBED_DIM=1024 → se recorta
        def _fake(textos, model, dim):
            assert dim == 1024  # el cap se pasa al proveedor
            return np.full((len(textos), 2560), 0.1, dtype=np.float32).tolist()

        monkeypatch.setitem(embed_mod._PROVIDERS, "fake", _fake)
        monkeypatch.setattr(embed_mod, "EMBED_PROVIDER", "fake")
        monkeypatch.setattr(embed_mod, "EMBED_DIM", 1024)
        vectores = embeddear(["a", "b"])
        assert len(vectores) == 2
        assert len(vectores[0]) == 1024

    def test_dim_explicito_supera_embed_dim(self, monkeypatch):
        def _fake(textos, model, dim):
            return np.full((len(textos), 5), 0.5, dtype=np.float32).tolist()

        monkeypatch.setitem(embed_mod._PROVIDERS, "fake", _fake)
        monkeypatch.setattr(embed_mod, "EMBED_PROVIDER", "fake")
        monkeypatch.setattr(embed_mod, "EMBED_DIM", 1024)
        vectores = embeddear(["a"], dim=3)
        assert len(vectores[0]) == 3

    def test_dim_none_por_defecto_usa_embed_dim(self, monkeypatch):
        # dim=None → cap por defecto EMBED_DIM (la puerta única siempre aplica el cap)
        def _fake(textos, model, dim):
            assert dim == 2048
            return _normalizar(np.full((len(textos), 4096), 0.5, dtype=np.float32))

        monkeypatch.setitem(embed_mod._PROVIDERS, "fake", _fake)
        monkeypatch.setattr(embed_mod, "EMBED_PROVIDER", "fake")
        monkeypatch.setattr(embed_mod, "EMBED_DIM", 2048)
        vectores = embeddear(["a"], dim=None)
        assert len(vectores[0]) == 2048

    def test_proveedor_ya_recorto_no_renormaliza_de_nuevo(self, monkeypatch):
        # el proveedor ya recortó (pista provider-side); el backstep no toca
        def _fake(textos, model, dim):
            return _normalizar(np.full((len(textos), dim), 0.5, dtype=np.float32))

        monkeypatch.setitem(embed_mod._PROVIDERS, "fake", _fake)
        monkeypatch.setattr(embed_mod, "EMBED_PROVIDER", "fake")
        monkeypatch.setattr(embed_mod, "EMBED_DIM", 1024)
        vectores = embeddear(["a"])
        assert len(vectores[0]) == 1024
        assert np.isclose(np.linalg.norm(vectores[0]), 1.0, atol=1e-5)


# =============== Seguridad de dimensiones (pipeline) ============

class TestSeguridadDimensiones:
    """
    _comprobar_dim() es la misma lógica que ejecuta ejecutar_pipeline() en la
    fase EMBED: la dim final del índice = min(dim del modelo, EMBED_DIM).
    EMBED_DIM (.env) nunca agranda; se recorta y renormaliza (embeddear()).
    """

    def test_dim_igual_ok_sin_cache(self):
        assert _comprobar_dim(1024, 1024, None) == ("ok", "")

    def test_dim_igual_ok_cache_por_bajo(self):
        # dim máxima declarada <= dim final: no hay información extra
        assert _comprobar_dim(1024, 1024, 1024) == ("ok", "")

    def test_embed_dim_menor_info(self):
        # EMBED_DIM 384 < dim máxima declarada 2560: se indexa a EMBED_DIM
        nivel, msg = _comprobar_dim(384, 384, 2560)
        assert nivel == "info"
        assert "máxima del modelo (2560)" in msg
        assert "384 dims (EMBED_DIM)" in msg

    def test_embed_dim_mayor_aviso(self):
        # EMBED_DIM 2560 > dim generada 1024: el índice se crea con la dim generada
        nivel, msg = _comprobar_dim(1024, 2560, None)
        assert nivel == "aviso"
        assert "es mayor que la dim generada (1024)" in msg
        assert "el índice se crea con 1024 dims" in msg

    def test_aviso_con_cache(self):
        nivel, _msg = _comprobar_dim(1024, 2560, 1024)
        assert nivel == "aviso"

    def test_corpus_vacio(self):
        assert _comprobar_dim(0, 1024, None) == ("vacio", "no se generaron embeddings (corpus vacío)")
