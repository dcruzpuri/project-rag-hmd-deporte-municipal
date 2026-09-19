"""Utilidades de formato compartidas (duraciones, etc.)."""
from __future__ import annotations


def fmt_dur(seg: float) -> str:
    """Formatea una duración en segundos como `h m s` con las unidades presentes.

    Reglas:
      - seg >= 3600   -> "1h 02m 03s" (siempre 2 dígitos en m/s para alineación)
      - 60 <= seg < 3600 -> "5m 07s"
      - 1 <= seg < 60  -> "45s"  (o "0.85s" si es fracción sub-segunda)
      - 0 < seg < 1    -> "0.85s"
      - seg <= 0       -> "0s"
    """
    if seg is None or seg <= 0:
        return "0s"
    if seg < 1:
        return f"{seg:.2f}s"
    seg_i = int(seg)
    h, r = divmod(seg_i, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    if s == seg_i:
        return f"{s}s"
    return f"{seg:.2f}s"
