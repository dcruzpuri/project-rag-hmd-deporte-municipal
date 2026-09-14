"""
src/informe.py
Genera el informe de indexación en markdown (por defecto ``output/informe_indexacion.md``)
al finalizar el pipeline: todos los parámetros aplicados y las métricas de la
ejecución (tiempos por fase, chunking, scoring, dedup, índice final) listas
para la toma de decisiones.

El pipeline ensambla el dict de métricas (``datos``); este módulo solo renderiza:
no lee config ni hace side effects de red, lo que permite testear la salida aislada.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FASES = ("PREFLIGHT", "LOAD", "CLEAN", "TAG", "CHUNK", "EMBED", "TSD", "INDEX", "FIN")


def _tabla(filas: list[tuple[str, str, str]]) -> str:
    """Renderiza filas (variable, valor, nota) como tabla markdown."""
    lineas = ["| Variable | Valor | Nota |", "|---|---|---|"]
    for var, valor, nota in filas:
        lineas.append(f"| `{var}` | {valor} | {nota} |")
    return "\n".join(lineas)


def formatear_duracion(segundos: float | None) -> str:
    """Duración legible: milisegundos para < 1 s, segundos (2 decimales) para el resto.

    ``None``, o < 0.05 ms (el mínimo que distingue la precisión de ms) → ``—``.
    Usada en el informe para no mostrar falsos 0.0 s.
    """
    if not segundos or segundos < 5e-05:
        return "—"
    if segundos < 1:
        return f"{segundos * 1000:.1f} ms"
    return f"{segundos:.2f} s"


def _fases_md(fases: dict[str, float], resumen: dict[str, str]) -> str:
    """Tabla de tiempos por fase con el resumen de cada una."""
    lineas = ["| Fase | Resultado | Tiempo |", "|---|---|---|"]
    for fase in _FASES:
        if fase in resumen:
            lineas.append(f"| {fase} | {resumen[fase]} | {formatear_duracion(fases.get(fase, 0))} |")
    return "\n".join(lineas)


def _scoring_md(scoring: dict[str, Any] | None, dedup: dict[str, Any] | None) -> list[str]:
    """Sección TSD (scoring + dedup); si TSD va desactivado, lo indica."""
    if scoring is None and dedup is None:
        return [
            "## 6. TSD (scoring + dedup)",
            "",
            "Bloque TSD desactivado (`TAG_SCORING_DEDUP=false`): sin scoring ni deduplicación.",
            "",
        ]
    lineas = ["## 6. TSD (scoring + dedup)", ""]
    if scoring is not None:
        lineas += [
            "### 6.1 Scoring (`semantic_score`)",
            "",
            "| Métrica | Valor | Interpretación |",
            "|---|---|---|",
            f"| mín | {scoring['semantic_score_min']} | peor chunk puntuado |",
            f"| media | {scoring['semantic_score_media']} | calidad media del corpus |",
            f"| máx | {scoring['semantic_score_max']} | mejor chunk del corpus |",
            (
                f"| chunks con score ≥ 0.6 | {scoring['score_buenos_n']} de {scoring['chunks_n']} ({round(scoring['score_buenos_pct'], 2)} %) | cuota de chunks 'útiles' según el modelo |"
                if scoring.get("score_buenos_n") is not None and scoring.get("chunks_n") is not None
                else f"| chunks con score ≥ 0.6 | {round(scoring['score_buenos_pct'], 2)} % | cuota de chunks 'útiles' según el modelo |"
            ),
            f"| centralidad media | {scoring['centralidad_media']} | cohesión del corpus (coseno vs. centroide) |",
            f"| redundancia media | {scoring['redundancia_media']} | similitud media con el vecino más cercano |",
            f"| tiempo | {formatear_duracion(scoring['tiempo_s'])} | fase SCORING |",
            "",
        ]
    if dedup is not None:
        lineas += [
            "### 6.2 Deduplicación",
            "",
            "| Métrica | Valor |",
            "|---|---|",
            f"| umbral coseno (`DEDUP_UMBRAL`) | {dedup['umbral']} | |",
            f"| chunks antes | {dedup['chunks_pre']} | |",
            f"| chunks después | {dedup['chunks_post']} | |",
            f"| descartados (política exacta) | {dedup['descartados_exactos']} | |",
            f"| descartados (semánticos) | {dedup['descartados_semantico']} | |",
            f"| total descartados | {dedup['descartados_total']} ({dedup['descartados_pct']} %) | |",
            f"| tiempo | {formatear_duracion(dedup.get('tiempo_s'))} | fase DEDUP |",
            "",
        ]
    return lineas


def _senales(datos: dict[str, Any]) -> list[str]:
    """Heurísticas automáticas sobre las métricas de la ejecución."""
    s: list[str] = []
    stats = datos.get("chunk_stats") or {}
    dedup = datos.get("dedup")
    dim_msg = (datos.get("dim_msg") or "").strip()
    preflight = datos.get("preflight") or {}
    scoring = datos.get("scoring")

    if stats.get("cortos"):
        s.append(
            f"**Chunking:** hay {stats['cortos']} chunks < 50 caracteres (ruido probable). "
            "Revisa los loaders o sube `CHUNK_SIZE`."
        )
    if dedup is not None and dedup.get("descartados_pct", 0) > 50:
        s.append(
            f"**Dedup:** se descartó el {dedup['descartados_pct']} % de los chunks (>50 %). "
            "Si es excesivo, baja `DEDUP_UMBRAL`; para más dedup, súbelo."
        )
    if dedup is not None and dedup.get("descartados_total", 0) == 0:
        s.append(
            "**Dedup:** no se descartó ningún chunk: `DEDUP_UMBRAL` puede estar "
            "demasiado alto o el corpus muy diverso."
        )
    if dim_msg:
        s.append(f"**Dimensiones:** {dim_msg}.")
    if preflight.get("verificado_por") == "cache_env":
        s.append(
            "**Preflight:** la disponibilidad se verificó por caché `.env` (sin comprobación "
            "online, p. ej. red cortada). La dim usada es fiable solo si "
            "`EMBED_DIM_MAX_*` está declarada."
        )
    if scoring is not None and scoring.get("score_buenos_pct", 100) < 50:
        s.append(
            f"Scoring: la distribución de `semantic_score` se concentra por debajo de 0.6 "
            f"({scoring.get('score_buenos_pct', 0)} % de chunks superan el umbral). Es una señal "
            "interna de priorización del corpus, no una métrica de precisión del retrieval; "
            "debe validarse con consultas reales. Si el corpus se nota pobre, revisa el "
            "modelo de TAG (solo ve los primeros 6000 caracteres de cada fuente)."
        )
    if not s:
        s.append("Sin señales de alerta: las métricas están dentro de rangos habituales.")
    return [f"- {x}" for x in s]


def generar_informe(datos: dict[str, Any], ruta: str | Path = "output/informe_indexacion.md") -> Path:
    """Escribe el informe markdown de indexación y devuelve su ruta.

    Args:
        datos: métricas y parámetros ensamblados por el pipeline:
            rutas, parametros (lista de (variable, valor, nota)), tiempo_total_s,
            fases (tiempos por fase), resumen_fases (texto por fase), num_documentos,
            num_chunks_pre_dedup/post_dedup, chunk_stats, dim_embedding, dim_msg,
            preflight, indice (nombre/vectores/spacer), scoring, dedup.
        ruta: ruta del informe (por defecto ``output/informe_indexacion.md``).

    Returns:
        Ruta absoluta del informe escrito.
    """
    ruta = Path(ruta)
    fases: dict[str, float] = datos.get("fases") or {}
    resumen_fases: dict[str, str] = datos.get("resumen_fases") or {}
    stats: dict[str, Any] = datos.get("chunk_stats") or {}
    indice: dict[str, Any] = datos.get("indice") or {}
    scoring = datos.get("scoring")
    dedup = datos.get("dedup")
    preflight = datos.get("preflight") or {}

    lineas: list[str] = [
        "# Informe de indexación",
        "",
        f"- **Fecha:** {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Corpus (rutas):** {', '.join(datos.get('rutas') or [])}",
        f"- **Tiempo total:** {formatear_duracion(datos.get('tiempo_total_s', 0))}",
        "",
        "## 1. Parámetros aplicados",
        "",
        _tabla(datos.get("parametros") or []),
        "",
        "## 2. Preflight",
        "",
        (f"- **Verificado por:** `{preflight.get('verificado_por', 'online')}`" if preflight else "- **Verificado por:** —"),
    ]
    if preflight and preflight.get("dim_modelo"):
        lineas.append(f"- **Dim máxima del modelo (`EMBED_DIM_MAX_*`):** {preflight['dim_modelo']}")
    lineas += [
        f"- **Aviso:** {preflight.get('aviso') or 'sin avisos'}",
        "",
        "## 3. Métricas por fase",
        "",
        _fases_md(fases, resumen_fases),
        "",
        "## 4. Chunking",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| min | {stats.get('min', 0)} |",
        f"| p25 | {stats.get('p25', 0)} |",
        f"| media | {stats.get('media', 0)} |",
        f"| p50 | {stats.get('p50', 0)} |",
        f"| p75 | {stats.get('p75', 0)} |",
        f"| max | {stats.get('max', 0)} |",
        f"| chunks cortos (< 50) | {stats.get('cortos', 0)} |",
        "",
        "## 5. Índice final (ChromaDB)",
        "",
        f"- **Colección:** `{indice.get('nombre', '—')}`",
        f"- **Vector space:** `{indice.get('space', '—')}` (coseno)",
        f"- **Dim de embedding:** {indice.get('dim', datos.get('dim_embedding', 0))}",
        f"- **Vectores insertados:** {indice.get('vectores_insertados', datos.get('num_chunks_post_dedup', 0))}",
        f"- **Vectores totales en la colección:** {indice.get('vectores_totales', '—')}",
        f"- **Recreado (`--recreate-index`):** {indice.get('recreado', False)}",
        "",
        *_scoring_md(scoring, dedup),
        "## 7. Señales y criterios de decisión",
        "",
        * _senales(datos),
    ]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta
