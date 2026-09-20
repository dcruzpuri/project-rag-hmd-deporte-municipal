"""
main.py

CLI del proyecto RAG HMD Deporte Municipal.

Comandos disponibles:
    python main.py --query "..."    # Retrieval puro (sin LLM)
    python main.py --ask "..."      # RAG completo (respuesta + fuentes)
    python main.py --index          # Indexa el corpus completo (offline)
    python main.py --prepare        # Alias de --index (compatibilidad con el enunciado)

Opciones:
    --top-k N            Override del TOP_K de config para --query y --ask
    --json               Salida en JSON (solo con --ask)
    --recreate-index     Borra la colección Chroma antes de indexar

Diseño: este archivo NO contiene lógica RAG.
Solo hace de CLI + llamada a los módulos de src/.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------


def _cmd_query(pregunta: str, top_k: int, as_json: bool) -> int:
    """Retrieval puro: muestra top-k chunks sin llamar al LLM."""
    from src.retrieve import recuperar

    try:
        chunks = recuperar(pregunta, top_k=top_k)
    except Exception as e:
        print(f"[ERROR] Fallo en retrieval: {e}", file=sys.stderr)
        return 1

    if not chunks:
        print("No se recuperó ningún chunk.")
        return 0

    if as_json:
        print(json.dumps(chunks, indent=2, ensure_ascii=False))
        return 0

    print(f"\n=== Retrieval para: {pregunta!r} (top_k={top_k}) ===\n")
    for i, c in enumerate(chunks, 1):
        dist = c.get("distance", 0.0)
        source = c.get("source", "desconocido")
        texto = (c.get("text") or "").strip()
        print(f"--- Chunk {i} | distancia={dist:.4f} | fuente={source} ---")
        print(texto[:400] + ("..." if len(texto) > 400 else ""))
        print()
    return 0


def _cmd_ask(pregunta: str, top_k: int, as_json: bool) -> int:
    """RAG completo: respuesta + fuentes + métricas."""
    from src.logic import responder

    resultado = responder(pregunta, top_k=top_k)

    if as_json:
        print(json.dumps(resultado, indent=2, ensure_ascii=False, default=str))
        return 0 if not resultado.get("error") else 1

    if resultado.get("error"):
        print(f"[ERROR] {resultado['error']}", file=sys.stderr)
        return 1

    print(f"\n=== Respuesta ===\n{resultado['respuesta']}\n")

    if resultado.get("fuentes"):
        print("=== Fuentes ===")
        for f in resultado["fuentes"]:
            print(f"  - {f}")
        print()

    if resultado.get("abstained"):
        print("[INFO] El sistema se abstuvo: no hay evidencia suficiente en el corpus.\n")

    metricas = resultado.get("metrics") or {}
    if metricas:
        print("=== Métricas ===")
        for k, v in metricas.items():
            print(f"  {k}: {v}")
        print()

    return 0


def _cmd_index(recreate: bool) -> int:
    """Indexa el corpus completo desde data/ hasta Chroma."""
    from config import CHROMA_DIR
    from src.pipeline import ejecutar_pipeline

    try:
        ejecutar_pipeline(
            ["data"],
            persist_dir=CHROMA_DIR,
            recreate_index=recreate,
        )
    except Exception as e:
        print(f"[ERROR] Fallo en indexación: {e}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Parser de argumentos
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RAG HMD Deporte Municipal — CLI",
    )

    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument(
        "--query",
        type=str,
        metavar="PREGUNTA",
        help="Retrieval puro: muestra top-k chunks sin LLM",
    )
    grupo.add_argument(
        "--ask",
        type=str,
        metavar="PREGUNTA",
        help="RAG completo: recupera contexto y genera respuesta con Gemini",
    )
    grupo.add_argument(
        "--index",
        action="store_true",
        help="Indexa el corpus completo en ChromaDB",
    )
    grupo.add_argument(
        "--prepare",
        action="store_true",
        help="Alias de --index (compatibilidad con el enunciado)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override del TOP_K de config (solo para --query y --ask)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Salida en JSON (para --query y --ask)",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Borra la colección Chroma antes de indexar (solo para --index)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # TOP_K efectivo: override del CLI o valor de config
    from config import TOP_K
    k = args.top_k if args.top_k is not None else TOP_K

    if k <= 0:
        print(f"[ERROR] --top-k debe ser > 0 (recibido: {k})", file=sys.stderr)
        return 2

    if args.query:
        return _cmd_query(args.query, k, args.json)

    if args.ask:
        return _cmd_ask(args.ask, k, args.json)

    if args.index or args.prepare:
        return _cmd_index(args.recreate_index)

    # No debería llegar aquí por el required=True, pero por seguridad:
    print("[ERROR] Debes indicar --query, --ask o --index", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())