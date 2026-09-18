"""
scripts/auditar_dedup.py
Muestreo bajo demanda de la auditoría de pares descartados (JSONL).

Lee la auditoría generada por el pipeline (DEDUP_AUDIT, por defecto
output/dedup_audit.jsonl), filtra por fuente y/o motivo y muestra los pares
descartado <-> ganador (o clave exacta) ordenados por similitud, sin tocar las
etapas TSD: responde «cómo muestreo los pares de juegos» sin reindexar.

Ejecutar desde la raíz del proyecto:
    python scripts/auditar_dedup.py --fuente 211549-0-juegos-deportivos-actual.txt --top 20
    python scripts/auditar_dedup.py --motivo clave --top 10
    python scripts/auditar_dedup.py --ruta output/dedup_audit.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEDUP_AUDIT_RUTA


def _normalizar_motivo(motivo: str | None) -> str | None:
    """Acepta `coseno`/`clave` (los de la doc) o el literal que escribe dedup."""
    if not motivo:
        return None
    return {"coseno": "dedup_coseno", "clave": "dedup_clave"}.get(
        motivo, f"dedup_{motivo}")


def _filas(ruta: str, fuente: str | None, motivo: str | None,
           top: int | None) -> list[tuple[float, dict]]:
    """Devuelve (sim, evento) filtrados por fuente (substring del chunk
    descartado) y por motivo, los descartes coseno ordenados por similitud
    descendente y los por clave al final (no tienen sim), con recorte opcional."""
    motivo_canon = _normalizar_motivo(motivo)
    filas: list[tuple[float, dict]] = []
    for linea in open(ruta, encoding="utf-8"):
        linea = linea.strip()
        if not linea:
            continue
        ev = json.loads(linea)
        if fuente and fuente not in str(ev.get("fuente", "")):
            continue
        if motivo_canon and ev.get("motivo") != motivo_canon:
            continue
        filas.append((ev.get("sim", float("-inf")), ev))
    filas.sort(key=lambda p: (-p[0], p[1].get("fuente", "")))
    return filas[:top] if top else filas


def _mostrar(filas: list[tuple[float, dict]]) -> None:
    for sim, ev in filas:
        if ev["motivo"] == "dedup_coseno":
            par = ev["pareja"]
            sim_txt = f"{sim:.4f}"
            print(f"{sim_txt}  `{ev['fuente']}` · chunk {ev.get('chunk_index')} (fila {ev.get('pos')})")
            print(f"    descartado: \"{ev['snip']}\"")
            print(f"         vira a: `{par['fuente']}` · chunk {par.get('chunk_index')} "
                  f"(fila {par.get('pos')}) · score {par['score']}")
            print(f"              \"{par['snip']}\"")
        else:  # dedup_clave: repetición literal descartada por clave exacta
            claves = ev.get("claves") or {}
            kvs = {k: v for k, v in claves.items() if v is not None}
            kvs_txt = " · ".join(f"{k}={v}" for k, v in sorted(kvs.items()))
            print(f"{'—':<6}  `{ev['fuente']}` · chunk {ev.get('chunk_index')} "
                  f"(fila {ev.get('pos')}) · {ev.get('policy')}: {kvs_txt}")
            print(f"    descartado: \"{ev['snip']}\"")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Muestra pares de la auditoría de dedup (sin tocar el pipeline)"
    )
    parser.add_argument("--fuente", default=None,
                        help="Filtro por fuente del chunk descartado (substring)")
    parser.add_argument("--motivo", default=None,
                        help="Filtro por motivo: coseno | clave (o dedup_coseno/dedup_clave)")
    parser.add_argument("--top", type=int, default=None,
                        help="Máximo de pares a mostrar")
    parser.add_argument("--ruta", default=DEDUP_AUDIT_RUTA,
                        help=f"Ruta del JSONL (por defecto: {DEDUP_AUDIT_RUTA})")
    args = parser.parse_args(argv)

    if not Path(args.ruta).is_file():
        print(f"auditoría no encontrada: {args.ruta} (¿se ejecutó el pipeline con DEDUP_AUDIT=true?)")
        return

    filas = _filas(args.ruta, args.fuente, args.motivo, args.top)
    if not filas:
        print(f"sin pares que coincidan (fuente={args.fuente or '—'} "
              f"motivo={args.motivo or '—'}) en {args.ruta}")
        return
    _mostrar(filas)
    print(f"({len(filas)} pares)")


if __name__ == "__main__":
    main()
