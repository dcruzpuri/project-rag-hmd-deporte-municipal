"""
scripts/embedded_questions.py
Genera un set de preguntas a partir de `output/embeddings.json`.

Origen de datos:
  - `output/embeddings.json` (export del pipeline, EXPORT_EMBEDDINGS=true):
    lista de registros `{"text", "metadata", "embedding"}`. Es el conjunto
    post-dedup que entró al índice. Es la fuente autoritativa: cada `text`
    genera `MAX_QUESTIONS` registros de salida (o 1 si el LLM no genera).

Activación:
  El script no hace nada si en `.env` no hay `QUESTION_SET=true`.
  `MAX_QUESTIONS` de `.env` fija cuántas preguntas (1..MAX_QUESTIONS) pide el
  LLM por texto.

Generación (una llamada por texto, en paralelo con `--workers`):
  Proveedor según `GEN_PROVIDER` de .env (ollama | huggingface | google).
  Temperatura de la llamada: 0.5 (fijada aquí para los 3 proveedores, no se
  lee de .env para mantener un solo lugar de cambio).
  Nº de preguntas por texto: `MAX_QUESTIONS` de .env (se pide al LLM y se
  usa para repartir los registros de salida).

  El LLM devuelve un array JSON de objetos planos
  `{"pregunta": "...", "respuesta_posible": "..."}` (uno por pregunta).
  El script:
    - valida el array y recorta a `MAX_QUESTIONS`
    - si la pregunta es "NO GENERADA" o vacía, el registro sale vacío
      (para auditar el texto)
  El LLM NO escribe `text` ni `metadata`: `source`, `file_hash` y
  `doc_category` salen de `embeddings.json` sin pérdida.

Salida:
  JSONL en `QUESTION_SET_RUTA` de .env (default `output/question_set.jsonl`).
  Una línea por registro:

  {
    "pregunta": "<pregunta generada por el LLM>",
    "respuesta_posible": "<respuesta sencilla generada por el LLM a raíz del texto>",
    "source": "<valor desde metadata>",
    "file_hash": "<valor desde metadata>",
    "doc_category": "<valor desde metadata>",
    "text_sha": "<sha1 del text (solo para audit/resume)>",
  }

Resume:
  Sin `--resume` el archivo se sobreescribe en cada ejecución (`w`).
  Con `--resume` se lee el JSONL existente, se extrae el conjunto de
  `text_sha` ya emitidos y solo se reintentan los textos pendientes (modo
  append). Si no queda nada pendiente, sale sin llamar al LLM.
  `--no-resume` se mantiene por compatibilidad del CLI: es un sinónimo
  de "regenerar por completo" (la conducta por defecto).

  Batch atómico por texto: los N registros de un mismo texto se escriben
  y se descargan en un solo flush, así un corte a mitad no deja un texto
  a medias.

Selección aleatoria (--random N):
  Se eligen aleatoriamente N textos del pool (post --exclude y --max)
  garantizando al menos 1 texto por cada file_hash del pool. Si N < nº de
  file_hash del pool, se procesan el nº de file_hash (la cobertura manda).
  Si N >= tamaño del pool, se procesa todo el pool (sin repetir textos).

Ejecutar desde la raíz del proyecto:
    python -m scripts.embedded_questions                # todo el corpus
    python -m scripts.embedded_questions --max 5        # solo los primeros 5
    python -m scripts.embedded_questions --random 50    # 50 textos aleatorios
                                                       # (min. 1 por file_hash)
    python -m scripts.embedded_questions --exclude h123,h456
        # omite los registros cuyo metadata.file_hash contenga h123 o h456
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    GEN_PROVIDER, GEN_MODEL,
    OLLAMA_BASE_URL,
    HF_DEVICE, HF_TOKEN,
    GOOGLE_API_KEY,
    MAX_QUESTIONS, QUESTION_SET, GEN_TIMEOUT,
    QUESTION_SET_RUTA,
)

# Temperatura fijada a 0.5 para el set de preguntas (según spec).
_TEMP = 2.5

# Caché del pipeline HF (es pesado cargarlo)
_HF_GEN = None


# --- Conversores por proveedor -----------------------------------------------

def _chat_ollama(prompt: str) -> str:
    import requests
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": GEN_MODEL,
            "stream": False,
            "options": {"temperature": _TEMP},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=GEN_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _chat_huggingface(prompt: str) -> str:
    global _HF_GEN
    from transformers import pipeline
    if _HF_GEN is None:
        kwargs: dict[str, Any] = {"device": HF_DEVICE}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        _HF_GEN = pipeline("text-generation", model=GEN_MODEL, **kwargs)
    out = _HF_GEN(prompt, do_sample=False, return_full_text=False)
    return out[0]["generated_text"]


def _chat_google(prompt: str) -> str:
    import requests
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no está definida en .env")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEN_MODEL}:generateContent",
        headers={"x-goog-api-key": GOOGLE_API_KEY},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": _TEMP,
                "maxOutputTokens": 4096,
            },
        },
        timeout=GEN_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


_CHAT: dict[str, Any] = {
    "ollama": _chat_ollama,
    "huggingface": _chat_huggingface,
    "google": _chat_google,
}


def _llm_call(prompt: str) -> str:
    fn = _CHAT.get(GEN_PROVIDER)
    if fn is None:
        raise ValueError(f"GEN_PROVIDER no soportado: {GEN_PROVIDER!r}")
    return fn(prompt)


# --- Prompt (según spec del usuario) ---------------------------------------

PROMPT = """
Eres un evaluador de sistemas RAG sobre un corpus de servicios de deporte municipal.

Tu tarea:
A partir del TEXTO que te doy, genera {max_q} preguntas, cada una con su
respuesta, que un usuario podría hacerle a un chatbot sobre este contenido.

Reglas estrictas:
1. Cada pregunta debe poder responderse ÚNICAMENTE con la información del TEXTO.
2. No inventes datos ni uses conocimiento externo.
3. Las preguntas deben ser claras, concretas y en español.
4. Evita preguntas triviales del tipo "¿de qué trata el texto?".
5. Prioriza preguntas sobre:
   - horarios, fechas y días de cierre
   - precios, descuentos y condiciones
   - instalaciones y equipamientos
   - normas, requisitos y procedimientos
   - transporte, ubicación y accesibilidad
6. Si el texto es muy corto o no tiene información útil, genera solo 1 pregunta
   o pon en su lugar "NO GENERADA" tanto en pregunta, como en respuesta.
7. Si el TEXTO se refiere a una instalación concreta (p. ej. una piscina o un
   polideportivo con nombre propio) y la pregunta va sobre ESA instalación
   (horarios, tarifas, normas, localización...), la pregunta DEBE incluir su
   nombre. No incluyas el nombre en estos dos casos:
   a) cuando la pregunta busca instalaciones (p. ej. cercanas a un lugar o
   dentro de un barrio/neighborhood), o
   b) cuando el nombre es precisamente lo que la pregunta está buscando
   (p. ej. "¿en qué instalación se celebra ...?", "¿dónde puedo bañarme
   al aire libre?").


Formato de salida (JSON válido, sin texto extra): un array con 1 a {max_q}
objetos, uno por pregunta:
[
  {
    "pregunta": "<pregunta 1>",
    "respuesta_posible": "<respuesta sencilla 1 a raíz del TEXTO>",
  },
  {
    "pregunta": "<pregunta 2>",
    "respuesta_posible": "<respuesta sencilla 2 a raíz del TEXTO>",
  }
]

Si se genera 1 pregunta, el array tiene 1 objeto.

TEXTO:
{text}
"""



# --- Parseo de la respuesta del LLM ----------------------------------------

# La respuesta esperada es un array de objetos planos
# `{"pregunta": ..., "respuesta_posible": ...}` (1..MAX_QUESTIONS). Si el
# LLM lo envuelve en texto o en un objeto, se busca el primer array JSON del
# interior. `_parse_respuesta` devuelve None si no hay array parseable
# (el worker lo cuenta como fallo); una lista vacía (array vacío explícito)
# se conserva para auditar el registro.

JSON_ARR = re.compile(r"\[[\s\S]*\]")


def _primera_lista(valor: Any) -> list[dict[str, Any]]:
    """Condiciona la respuesta del LLM a 'lista de objetos'. Si viene un objeto
    suelto (desviación del formato), lo envuelve; si no, []."""
    if isinstance(valor, dict):
        return [valor]
    if isinstance(valor, list):
        return [o for o in valor if isinstance(o, dict)]
    return []


def _parse_respuesta(raw: str, limite: int) -> list[tuple[str, str]] | None:
    """Extrae los pares (pregunta, respuesta_posible) de la respuesta cruda.

    `limite` recorta a `MAX_QUESTIONS`. El valor "NO GENERADA" (o vacío) se
    normaliza a cadena vacía. Devuelve None si la respuesta no contiene un
    array JSON utilizable.
    """
    m = JSON_ARR.search(raw)
    if not m:
        return None
    data = json.loads(m.group(0))
    if isinstance(data, dict) and "preguntas" in data:
        data = data["preguntas"]
    pares: list[tuple[str, str]] = []
    for obj in _primera_lista(data)[:limite]:
        q = str(obj.get("pregunta") or "").strip()
        r = str(obj.get("respuesta_posible") or "").strip()
        if q == "NO GENERADA":
            q = ""
        if r == "NO GENERADA":
            r = ""
        pares.append((q, r))
    return pares


# --- Worker (una llamada LLM por texto) ------------------------------------

def _sha1(texto: str) -> str:
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()


def _procesar(pos: int, texto: str, metadata: dict) -> list[dict[str, Any]]:
    prompt = PROMPT.replace("{max_q}", str(MAX_QUESTIONS)).replace("{text}", texto)
    raw = _llm_call(prompt)
    pares = _parse_respuesta(raw, MAX_QUESTIONS)
    if pares is None:
        raise ValueError(f"respuesta LLM sin array JSON @{pos}: {raw[:200]!r}")
    if not pares:
        # Textos sin pregunta generada: se deja el registro vacío para auditar.
        pares = [("", "")]
    meta = metadata or {}
    text_sha = _sha1(texto)
    return [{
        "pregunta": q,
        "respuesta_posible": r,
        "source": meta.get("source", ""),
        "file_hash": meta.get("file_hash", ""),
        "doc_category": meta.get("doc_category", ""),
        "text_sha": text_sha,
    } for q, r in pares]


def _lee_text_sha_existente(ruta: Path) -> tuple[set[str], bool]:
    """Con `--resume`: conjunto de `text_sha` ya emitidos en el JSONL.

    Devuelve `(vistos, stale)`: `stale=True` si algún registro no tiene
    `text_sha` (salida de una versión anterior; no se puede reanudar de
    ahí y se sobreescribe)."""
    vistos: set[str] = set()
    stale = False
    if not ruta.exists():
        return vistos, stale
    try:
        with ruta.open(encoding="utf-8") as f:
            for linea in f:
                try:
                    reg = json.loads(linea)
                except Exception:
                    continue
                t = reg.get("text_sha")
                if t:
                    vistos.add(t)
                else:
                    stale = True
    except Exception:
        return vistos, True
    return vistos, stale


# --- Selección aleatoria --------------------------------------------------------

def _seleccion_aleatoria(
    textos: list[str],
    file_hashes: list[str],
    n: int,
    rng: random.Random | None = None,
) -> list[int]:
    """Indices de `textos` para procesar con `--random n`.

    Garantías:
    - cada `file_hash` del pool aparece al menos una vez si n >= nº de hashes;
    - cada índice aparece una sola vez (sin repetir textos);
    - si n < nº de hashes, n se sube al nº de hashes (la cobertura manda).
    """
    rng = rng or random
    pool = len(textos)
    if n >= pool:
        return list(range(pool))

    # 1 texto aleatorio por file_hash (la cobertura manda por delante de n).
    por_hash: dict[str, list[int]] = {}
    for i, file_hash in enumerate(file_hashes):
        por_hash.setdefault(file_hash, []).append(i)
    elegidos = [rng.choice(grupo) for grupo in por_hash.values()]

    n_elegidos = min(n, pool)
    restantes = [i for i in range(pool) if i not in set(elegidos)]
    rng.shuffle(restantes)
    elegidos.extend(restantes[: max(0, n_elegidos - len(elegidos))])
    rng.shuffle(elegidos)
    return elegidos


# --- Main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera el set de preguntas a partir de output/embeddings.json.",
    )
    parser.add_argument(
        "--embeddings", default="output/embeddings.json",
        help="Ruta del dump del pipeline (default: output/embeddings.json)",
    )
    parser.add_argument(
        "--output", default=QUESTION_SET_RUTA,
        help=f"Ruta JSONL de salida (default: {QUESTION_SET_RUTA})",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Nº de hilos de LLM en paralelo (default: 4)",
    )
    parser.add_argument(
        "--max", type=int, default=0,
        help="Límite de textos a generar (0 = todos; útil para probar)",
    )
    parser.add_argument(
        "--random", type=int, default=0,
        help="Selecciona aleatoriamente N textos del pool garantizando mínimo 1 por "
             "file_hash distinto del pool (0 = secuencial; si N < nº de file_hash, "
             "se procesa ese nº)",
    )
    parser.add_argument(
        "--exclude", default="",
        help="file_hash a excluir, separados por coma (coincidencia parcial); "
             "se aplican a metadata.file_hash de embeddings.json",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reanudar: solo generar los textos no emitidos aún (modo append)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="No usar; equivalente al comportamiento por defecto (sobreescribir)",
    )
    args = parser.parse_args()

    # Gate de activación
    if str(QUESTION_SET).strip().lower() not in {"true", "1", "yes"}:
        print("QUESTION_SET no está activado en .env (debe ser 'true'). "
              "Salir sin hacer nada.")
        sys.exit(0)

    emb_ruta = Path(args.embeddings)
    if not emb_ruta.exists():
        print(f"embeddings.json no está en {emb_ruta}; no se generará nada. "
              f"(Ejecutar primero el pipeline con EXPORT_EMBEDDINGS=true)")
        sys.exit(1)

    print("Cargando embeddings.json ...")
    t0 = time.time()
    with emb_ruta.open(encoding="utf-8") as f:
        registros = json.load(f)
    n = len(registros)
    print(f"  {n} registros en {time.time() - t0:.1f}s")


    exclude = {t.strip() for t in args.exclude.split(",") if t.strip()}
    textos: list[str] = []
    metas: list[dict] = []
    excluidos = 0
    for r in registros:
        meta = dict(r.get("metadata") or {})
        file_hash = str(meta.get("file_hash") or "")
        if any(tok in file_hash for tok in exclude):
            excluidos += 1
            continue
        textos.append(str(r.get("text", "")))
        metas.append(meta)
    pool_n = len(textos)
    if exclude:
        print(f"  --exclude: {excluidos} registros excluidos por file_hash; "
              f"{pool_n} quedan")
    if args.max > 0:
        pool_n = min(pool_n, args.max)
        print(f"  --max={args.max}: procesando solo los primeros {pool_n}")

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)

    if pool_n == 0:
        print("[EMB-Q] 0 registros pendientes (vacíos o excluidos por "
              "--exclude); nada que hacer. Salir")
        sys.exit(0)

    pendientes: list[int] = list(range(pool_n))
    if args.random > 0:
        file_hashes_pool = [
            str(metas[i].get("file_hash") or "") for i in range(pool_n)
        ]
        antes = pool_n
        pendientes = _seleccion_aleatoria(
            textos[:pool_n], file_hashes_pool, args.random,
        )
        aviso = (
            " (N subido: nº de file_hash por delante)"
            if len(pendientes) > args.random else ""
        )
        print(f"  --random={args.random}: {antes} -> {len(pendientes)} textos "
              f"aleatorios{aviso}")
    modo_append = False
    if args.resume and salida.exists():
        vistos, stale = _lee_text_sha_existente(salida)
        if not stale:
            antes = len(pendientes)
            pendientes = [i for i in pendientes
                          if _sha1(textos[i]) not in vistos]
            print(f"  Resume: {antes - len(pendientes)} textos ya emitidos; "
                  f"{len(pendientes)} pendientes")
            modo_append = True
        else:
            print(f"  Resume: {salida} trae registros de una versión anterior "
                  f"(sin text_sha); se sobreescribe y se genera todo.")
    if not pendientes:
        print(f"[EMB-Q] todo emitido en {salida}; nada pendiente. Salir")
        sys.exit(0)
    ok_lines = 0
    t1 = time.time()
    fallos: list[tuple[int, str]] = []
    workers = max(1, min(args.workers, len(pendientes)))
    with ThreadPoolExecutor(max_workers=workers) as pool, \
            salida.open("a" if modo_append else "w", encoding="utf-8") as fh:
        futures = {pool.submit(_procesar, i, textos[i], metas[i]): i
                   for i in pendientes}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                regs = fut.result()
            except Exception as exc:
                fallos.append((i, f"{type(exc).__name__}: {exc}"))
                print(f"  fallo @{i}: {type(exc).__name__}: {exc}")
                continue
            for reg in regs:
                fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
            fh.flush()
            ok_lines += len(regs)
            hecho = ok_lines + len(fallos)
            if hecho % 50 == 0:
                print(f"  progreso: {hecho}/{len(pendientes)} textos "
                      f"({ok_lines} líneas) [{time.time() - t1:.0f}s]")

    t2 = time.time() - t1
    print(f"[EMB-Q] generado {ok_lines} registros en {t2:.0f}s "
          f"({len(fallos)} fallos de texto) -> {salida}")
    if fallos:
        for i, err in fallos[:10]:
            print(f"  detalle fallo @{i}: {err}")


if __name__ == "__main__":
    main()
