"""
src/embed.py
Convierte texto en vectores. Soporta 3 proveedores:
  - ollama       (offline)  -> POST /api/embed
  - huggingface  (online)   -> sentence-transformers
  - google       (online)   -> REST batchEmbedContents
Condición RAG: el índice y la consulta deben usar el MISMO modelo [5].
Garantía de dimensión: `embeddear()` nunca devuelve más de `EMBED_DIM`
dimensiones (corte del prefijo + renormalización); al ser la puerta única,
la garantía la heredan el índice y la consulta futura.
"""
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import requests
from huggingface_hub import repo_exists
from sentence_transformers import SentenceTransformer

from config import (
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBED_MODEL,
    EMBED_TIMEOUT,
    EMBED_PROVIDER,
    GOOGLE_API_KEY,
    GOOGLE_EMBED_MODEL,
    HF_DEVICE,
    HF_EMBED_MODEL,
    HF_TOKEN,
    OLLAMA_BASE_URL,
)

# Caché: SentenceTransformer se carga una sola vez por modelo (evita `global`)
_HF_CACHE: dict[str, SentenceTransformer] = {}

# Firma de los proveedores: (textos, modelo, máximo de dim) -> vectores
_FN_PROVEEDOR = Callable[[list[str], str | None, int | None], list[list[float]]]

# Cada cuantos lotes se imprime una línea de progreso (1 = cada lote).
_EMBED_LOG_EVERY = 10


def _log_progreso_lote(lote: int, total: int, textos: int, t_lote: float) -> None:
    """Log por lote con la marca de tiempo/fase de consola:
    `AAAAMMDD hh:mm:ss [EMBED]   lote i/N (n embeddings, s s) (Restante: 99m 99s)`.
    La ETA parte del tiempo del último lote (basta para pantalla)."""
    eta = max(0.0, t_lote * (total - lote))
    m, s = divmod(int(eta), 60)
    print(time.strftime("%Y-%m-%d %H:%M:%S") +
          f" [EMBED]   lote {lote}/{total} ({textos} embeddings, {t_lote:.1f}seg/lote) "
          f"(Restante: {m}m {s}s)")


def _embed_ollama(textos: list[str], model: str | None = None,
                  dim: int | None = None) -> list[list[float]]:
    """Ollama local: endpoint batch /api/embed (acepta varios inputs a la vez)."""
    session = requests.Session()
    url = f"{OLLAMA_BASE_URL}/api/embed"
    vectores: list[list[float]] = []
    n_lotes = (len(textos) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for i, inicio in enumerate(range(0, len(textos), EMBED_BATCH_SIZE), start=1):
        batch = textos[inicio:inicio + EMBED_BATCH_SIZE]
        payload: dict[str, Any] = {"model": model or EMBED_MODEL, "input": batch}
        if dim is not None:
            # Ollama corta al máximo del modelo si dim lo supera y lo ignora
            payload["dimensions"] = dim
        t_lote = time.time()
        resp = session.post(url, json=payload, timeout=EMBED_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" not in data:
            raise RuntimeError(
                "Ollama no devolvió 'embeddings': el endpoint batch /api/embed "
                "requiere Ollama >= 0.9. Actualizad Ollama o usad "
                "EMBED_PROVIDER=huggingface."
            )
        vectores.extend(data["embeddings"])
        if i % _EMBED_LOG_EVERY == 0 or i == n_lotes:
            _log_progreso_lote(i, n_lotes, len(batch), time.time() - t_lote)
    return vectores


def _embed_huggingface(textos: list[str], model: str | None = None,
                       dim: int | None = None) -> list[list[float]]:
    """HuggingFace vía sentence-transformers (modelo descargado del HF Hub)."""
    nombre = model or HF_EMBED_MODEL
    if nombre not in _HF_CACHE:  # solo si el proveedor es huggingface
        kwargs: dict[str, str] = {"device": HF_DEVICE}
        if HF_TOKEN:  # solo para modelos gated
            kwargs["token"] = HF_TOKEN
        _HF_CACHE[nombre] = SentenceTransformer(nombre, **kwargs)
    # normalize_embeddings=True: vectores unitarios, coherente con la métrica de coseno de Chroma
    # truncate_dim: slice del prefijo (MRL-first para modelos Matryoshka); None si dim >= dim nativa
    # encode() hace una única llamada (batch internal del backend): se avisa de
    # la única llamada y se mide el tiempo total real al terminar.
    t0 = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S") +
          f" [EMBED]   huggingface: {len(textos)} embeddings en una única llamada "
          f"(batch_size={EMBED_BATCH_SIZE})")
    vecs = _HF_CACHE[nombre].encode(textos, batch_size=EMBED_BATCH_SIZE,
                                    normalize_embeddings=True, truncate_dim=dim)
    out = vecs.tolist()
    print(time.strftime("%Y-%m-%d %H:%M:%S") +
          f" [EMBED]   huggingface: {len(out)} vectores en {time.time() - t0:.1f} s")
    return out


def _reintentar_cuota(url: str, headers: dict, body: dict) -> requests.Response:
    """POST con retry sobre cuota (429): espera según respuesta 'RetryInfo.retryDelay' 
    de la API como delay.
    El free tier de Gemini limita a 100 embeddings/min por modelo; una tanda
    que no cabe en la ventana se reintenta tras el delay que pide la respuesta
    (p. ej. "51s"). 15 intentos en total (el primero + 14 reintentos).
    """
    for intento in range(15):  # 200 directo o 14 reintentos sobre cuota
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        if resp.status_code != 429 or intento == 14:
            return resp
        delay = 60.0  # por defecto: la ventana de cuota dura ~1 min
        try:
            for d in resp.json().get("details", []):
                rd = d.get("retryDelay")
                if rd:
                    delay = float(str(rd).rstrip("s"))
                    break
        except (ValueError, KeyError, UnicodeDecodeError, TypeError):
            pass
        time.sleep(delay + 0.5)
    raise AssertionError("inaccesible: repetidos fallos de la llamada.")


def _embed_google(textos: list[str], model: str | None = None,
                  dim: int | None = None) -> list[list[float]]:
    """Google Gemini vía REST (requiere GOOGLE_API_KEY).

    `dim` se acepta y se ignora a propósito: el body de batchEmbedContents
    usado aquí no soporta `outputDimensionality` de forma fiable, así que el
    máximo de dimensiones lo aplica el recorte de `embeddear()`.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no está definida en .env")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model or GOOGLE_EMBED_MODEL}:batchEmbedContents")
    headers = {"x-goog-api-key": GOOGLE_API_KEY}
    n_modelo = f"models/{model or GOOGLE_EMBED_MODEL}"
    vectores: list[list[float]] = []
    n_lotes = (len(textos) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for lote, inicio in enumerate(range(0, len(textos), EMBED_BATCH_SIZE), start=1):
        batch = textos[inicio:inicio + EMBED_BATCH_SIZE]
        body = {"requests": [
            {"model": n_modelo, "content": {"parts": [{"text": t}]}}
            for t in batch
        ]}
        t_lote = time.time()
        resp = _reintentar_cuota(url, headers, body)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Google API {resp.status_code} en lote {lote} "
                f"({len(batch)} embeddings): {resp.text[:500]}"
            )
        # La API devuelve "values" (o "value" en algunas versiones) por embedding
        for r in resp.json()["embeddings"]:
            emb = r.get("values") or r.get("value")
            if emb is None:
                raise RuntimeError(
                    "Google API no devolvió 'values' ni 'value' por embedding: "
                    f"respuesta parcial: {r!r}"
                )
            vectores.append(emb)
        if lote % _EMBED_LOG_EVERY == 0 or lote == n_lotes:
            _log_progreso_lote(lote, n_lotes, len(batch), time.time() - t_lote)
    return vectores


# Dispatch por proveedor: añadir un proveedor nuevo = añadir una función y una línea
_PROVIDERS: dict[str, _FN_PROVEEDOR] = {
    "ollama": _embed_ollama,
    "huggingface": _embed_huggingface,
    "google": _embed_google,
}


def _normalizar(vectores: list[list[float]]) -> list[list[float]]:
    E = np.asarray(vectores, dtype=np.float32)
    normas = np.linalg.norm(E, axis=1, keepdims=True)
    normas[normas == 0] = 1.0  # evitar división por cero
    return (E / normas).tolist()


def _corte_dim(vectores: list[list[float]], dim_cap: int | None) -> list[list[float]]:
    """Tamaño máximo de los vectores: nunca supera `dim_cap`.

    Corta el prefijo (las primeras `dim_cap` coordenadas) y normaliza,
    coherente con la métrica coseno de Chroma. Es no-op si `dim_cap` es
    `None` o los vectores entran por dimensionalidad.
    """
    if dim_cap is None or not vectores or len(vectores[0]) <= dim_cap:
        return vectores
    return _normalizar([v[:dim_cap] for v in vectores])


def embeddear(textos: list[str], model: str | None = None,
              dim: int | None = None) -> list[list[float]]:
    """Convierte una lista de embeddings en vectores usando el proveedor de 'EMBED_PROVIDER'.

    Args:
        textos: embeddings a vectorizar.
        model: modelo concreto (por defecto, el de .env para el proveedor activo).
        dim: tope de dimensión; por defecto 'EMBED_DIM' de .env.

    Returns:
        Vectores normal cuya dimensión NO supera 'dim' (corte del prefijo
        + renormalización si el modelo produce más). Al ser la puerta única
        por la que pasan el índice y la consulta, la dim es consistente en ambas.
    """
    cap = dim if dim is not None else EMBED_DIM
    fn = _PROVIDERS.get(EMBED_PROVIDER)
    if fn is None:
        raise ValueError(f"EMBED_PROVIDER no soportado: {EMBED_PROVIDER!r}")
    # 1) pista al proveedor (Ollama/HF la cumplen en servidor); 2) recorte
    #    cliente (la garantía real, agnóstica de proveedor)
    return _corte_dim(fn(textos, model, cap), cap)


def embeddear_consulta(pregunta: str, model: str | None = None) -> list[float]:
    """Embed de una sola consulta (fase online / retrieval). MISMO modelo que al indexar [5]."""
    return embeddear([pregunta], model)[0]


def exportar_json(chunks: list, embeddings: list[list[float]] | None = None,
                  ruta: str = "output/embeddings.json") -> None:
    """Persiste text+metadata+embedding para inspección/debug."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    registros = []
    for i, chunk in enumerate(chunks):
        reg = {"text": chunk.page_content, "metadata": chunk.metadata}
        if embeddings is not None and i < len(embeddings):
            reg["embedding"] = embeddings[i]
        registros.append(reg)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2, default=str)


# --- Preflight: comprobar que EMBED_MODEL existe en EMBED_PROVIDER antes de empezar ----------

# Cortes de proveedor para el nombre de variable de caché (.env):
# EMBED_DIM_MAX_OLLAMA / EMBED_DIM_MAX_HF / EMBED_DIM_MAX_GOOGLE
_PROVEEDOR_ABREV: dict[str, str] = {"ollama": "OLLAMA", "huggingface": "HF", "google": "GOOGLE"}


def _dim_max_cache(proveedor: str, meta: dict[str, str]) -> int | None:
    """Caché offline opcional (`.env`): `EMBED_DIM_MAX_<proveedor>` (short name).
    Declara el máximo de dimensiones que maneja el modelo cuando la
    comprobación no puede hacerse online (red cortada / sin API key)."""
    clave = f"EMBED_DIM_MAX_{_PROVEEDOR_ABREV[proveedor]}"
    raw = meta.get(clave) or os.environ.get(clave)
    if raw is None or str(raw).strip() == "":
        return None
    return int(str(raw).strip())


def _pre_online(proveedor: str, modelo: str, meta: dict[str, str]) -> dict[str, Any] | None:
    """
    Comprobación online de disponibilidad del modelo.
    Returns dict con "disponible" (y opcionalmente "dim_modelo"), o None cuando el proveedor 
    no se puede comprobar online (red cortada, Ollama apagado, sin API key) y hay que caer 
    al caché de `.env`.
    """
    if proveedor == "ollama":
        base = meta.get("OLLAMA_BASE_URL") or OLLAMA_BASE_URL
        try:
            r = requests.get(f"{base}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.ConnectionError:
            return None  # Ollama no arrancado / red local sin servidor
        except requests.HTTPError as e:
            # 5xx = fallo del servidor (se asume vía caché); 4xx = no disponible
            if e.response is not None and 500 <= e.response.status_code < 600:
                return None
            return {"disponible": False}
        modelos = [m.get("name", "") for m in r.json().get("models", [])]
        # Ollama lista "namespace:tag" (p. ej. "qwen3-embedding:4b");
        # "sin tag" y "tag" son la misma entrada.
        nombre, _, _tag = modelo.partition(":")
        return {"disponible": any(m == modelo or m == nombre for m in modelos)}
    if proveedor == "huggingface":
        token = meta.get("HF_TOKEN") or HF_TOKEN
        try:
            # `repo_exists()` ya distingue por sí solo: repo no existe => `False`,
            # gateado sin acceso => `TRue` (existe). Excepciones = red cortada => se usa el caché.
            return {"disponible": repo_exists(modelo, token=token)}
        except (requests.RequestException, OSError, ValueError):
            return None  # red cortada: se usa el caché de .env
    if proveedor == "google":
        api_key = meta.get("GOOGLE_API_KEY") or GOOGLE_API_KEY
        if not api_key:
            return None  # sin API key => se usa el caché
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        try:
            r = requests.get(url, headers={"x-goog-api-key": api_key}, timeout=10)
            r.raise_for_status()
        except requests.RequestException:
            return None
        ids = [m.get("name", "").split("/")[-1] for m in r.json().get("models", [])]
        return {"disponible": any(modelo == m for m in ids)}
    # Proveedor no soportado (`config._resolver` ya lo rechazó, pero por si acaso)
    raise ValueError(f"PREFLIGHT: EMBED_PROVIDER no soportado: {proveedor!r}")


class PreflightResultado(TypedDict):
    """Resultado del preflight de disponibilidad del modelo."""
    disponible: bool
    dim_modelo: int | None
    verificado_por: str
    aviso: str | None


def verificar_modelo_disponible(metadatos: dict[str, str] | None = None) -> PreflightResultado:
    """
    Preflight: comprueba que EMBED_MODEL está disponible en EMBED_PROVIDER
    ANTES de empezar el pipeline.

    Args:
        metadatos: valores en bruto (p. ej. para tests). Sin argumentos se leen
            las constantes de config.py (ya resueltas de .env).

    Returns:
        dict con:
            "disponible": bool (siempre True; si no, se lanza RuntimeError).
            "dim_modelo": int | None — dimensión máxima declarada vía
                EMBED_DIM_MAX_*. None = el proveedor no la expone.
            "verificado_por": str — "online" | "cache_env".
            "aviso": str | None — comprobación degradada (red cortada / sin
                API key): se asume disponible vía caché de `.env`.

    Raises:
        RuntimeError: si el modelo NO está disponible (el pipeline debe abortar).
    """
    meta: dict[str, str] = {} if metadatos is None else dict(metadatos)
    proveedor = meta.pop("EMBED_PROVIDER", None) or EMBED_PROVIDER
    modelo = meta.pop("EMBED_MODEL", None) or EMBED_MODEL
    cache_dim = _dim_max_cache(proveedor, meta)

    resultado = _pre_online(proveedor, modelo, meta)
    if resultado is not None and not resultado["disponible"]:
        raise RuntimeError(
            f"PREFLIGHT: el modelo {modelo!r} NO está disponible en {proveedor!r}. "
            "Comprueba el nombre del modelo o que el servicio esté arrancado."
        )
    verificado = "online" if resultado is not None else "cache_env"
    dim_modelo = (resultado or {}).get("dim_modelo") or cache_dim

    if verificado == "cache_env":
        clave_cache = f"EMBED_DIM_MAX_{_PROVEEDOR_ABREV[proveedor]}"
        aviso = (
            f"PREFLIGHT: no se pudo verificar online {modelo!r} (red cortada / "
            "sin API key). Se asume disponible vía caché de `.env` como funcionamiento "
            "degradado."
            + (f" ({clave_cache}={dim_modelo})" if dim_modelo else f" ({clave_cache} no definido)")
        )
    else:
        aviso = None
    return {
        "disponible": True,
        "dim_modelo": dim_modelo,
        "verificado_por": verificado,
        "aviso": aviso,
    }