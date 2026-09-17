"""
src/informe.py
Genera el informe de indexación en markdown (por defecto
``output/informe_index_<guid8_chroma>_aaaaMMdd_hhmm.md``)
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


_GLOSARIO_TSD = [
    "> Definiciones de métricas (los vectores están normalizados: norma L2 = 1, métrica coseno):",
    "> - `semantic_score = 0.45·relevancia_LLM + 0.20·centralidad + 0.20·(1 − redundancia) + 0.15·autoridad`, recortado a [0, 1]. Señal interna de priorización del corpus, **no** una medida de calidad del chunk ni de relevancia frente a consultas.",
    "> - `centralidad`: coseno del chunk contra el centroide de la colección (media de todos los vectores). Coherencia con el centro del corpus.",
    "> - `redundancia`: coseno del chunk con su vecino más cercano (FAISS sobre los vectores normalizados; se excluye el propio). Alta redundancia media puede ser estructural (plantillas/CSVs de hechos) y no indicar corpus roto.",
    "> - **Política exacta**: chunk cuya metadata `dedup_policy` es `exact_key`, `group_only` o `exact_only` (decidida por el asesor de CSVs); se deduplica **por clave** (nunca por coseno): la repetición literal se descarta, los semánticos cercanos no.",
]


def _fuentes_md(fuentes: list[tuple[str, dict[str, int]]]) -> list[str]:
    """Tabla fuentes -> (docs, chunks pre, chunks post): los tres niveles del corpus.

    ``fuentes`` es ``sorted(fuentes.items())`` (orden alfabético determinista).
    Con corpus vacío (sin datos volcados) no se renderiza.
    """
    if not fuentes:
        return []
    lineas = ["| fuente | docs | chunks pre | chunks post (dedup) |", "|---|---|---|---|"]
    tot_docs = tot_pre = tot_post = 0
    for fuente, f in fuentes:
        lineas.append(f"| {fuente} | {f['docs']} | {f['chunks_pre']} | {f['chunks_post']} |")
        tot_docs += f["docs"]
        tot_pre += f["chunks_pre"]
        tot_post += f["chunks_post"]
    lineas.append(f"| **total** | {tot_docs} | {tot_pre} | {tot_post} |")
    vacias = [fuente for fuente, f in fuentes if f["chunks_pre"] and not f["chunks_post"]]
    if vacias:
        lineas.append(f"> **Fuente vacía post-dedup (crítica):** {', '.join(vacias)}")
    return lineas


def _redundancia_politica_md(scoring: dict[str, Any]) -> list[str]:
    """Tabla de redundancia por política de dedup (6.1); opcional: si no hay
    datos volcados, no se muestra. El umbral se toma del volcado del scoring
    (``redundancia_umbral`` = DEDUP_UMBRAL), no hardcodeado."""
    if scoring.get("redundancia_umbral") is None:
        return []
    umbral = scoring["redundancia_umbral"]
    por_pol = scoring.get("redundancia_por_politica") or {}
    if not por_pol:
        return []
    lineas = [
        f"Redundancia (similitud con el vecino más cercano) por política de deduplicación (umbral `{umbral}`):",
        "",
        "| política | n | media | p50 | p90 | % ≥ umbral |",
        "|---|---|---|---|---|---|",
    ]
    for pol in sorted(por_pol):
        r = por_pol[pol]
        lineas.append(f"| {pol} | {r['n']} | {r['media']} | {r['p50']} | {r['p90']} | {r['pct_sup_umbral']} % |")
    lineas += [
        "",
        "> La alta media de `group_only`/`exact_key` es estructural (plantillas y entidades de CSV): esas políticas se dedup por clave y nunca por coseno.",
        "",
    ]
    return lineas


def _cobertura_md(dedup: dict[str, Any] | None) -> list[str]:
    """Sección 6.3 — cobertura por categoría (pre/post dedup) y top-3 de tags.

    Solo se renderiza si TSD está activo y se volcó la cobertura; si no, no se
    muestra (la multi-facialidad no existe sin etiquetado).
    """
    if dedup is None or "chunks_por_categoria" not in dedup:
        return []
    cov = dedup["chunks_por_categoria"]
    if not cov:
        return []
    lineas = [
        "### 6.3 Cobertura por categoría",
        "",
        "| categoría | pre | post (dedup) |",
        "|---|---|---|",
    ]
    # La taxonomía cerrada ordena las 6 categorías fijas: si no están en el
    # corpus, aparecen explícitamente como 0 (no se ocultan).
    for cat in sorted(cov):
        lineas.append(f"| {cat} | {cov[cat]['pre']} | {cov[cat]['post']} |")
    top3 = dedup.get("tags_top3_post") or []
    lineas.append(
        f"> **top 3 tags (post):** "
        f"{' · '.join(f'{t} ({n})' for t, n in top3) if top3 else '—'}"
    )
    lineas.append(
        "> La categoría es única por fuente (la intención dominante según TAG); los tags son multi-faces (un chunk puede llevar varios). Limitación de TAG: solo ve el primer documento de cada fuente (≤ 6000 caracteres), que decide categoría, tags y relevancia."
    )
    lineas.append("")
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
        plural = "chunks" if stats["cortos"] != 1 else "chunk"
        s.append(
            f"**Chunking:** hay {stats['cortos']} {plural} < 50 caracteres (ruido "
            "probable). Revisa los loaders o sube `CHUNK_SIZE`."
        )
    # Alerta crítica (fuente a 0 post-dedup): si un archivo entero desaparece
    # del índice, su contenido deja de ser recuperable. No la ocultan las
    # heurísticas habituales: va explícita para que no se pase por alto.
    for fuente, f in (datos.get("fuentes") or []):
        pre, post = f.get("chunks_pre", 0), f.get("chunks_post", 0)
        if pre > 0 and post == 0:
            s.append(
                f"**Fuente vacía (crítica):** `{fuente}` quedó con 0 chunks "
                f"post-dedup ({pre} pre): su contenido ya no se recupera. "
                "Revisa `DEDUP_UMBRAL` o regenera el índice; con una única "
                "fuente de 1 chunk el descarte suele ser azar de la dedup "
                "semántica (el LLM de TAG varía entre ejecuciones)."
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
    cov = (dedup or {}).get("chunks_por_categoria") or {}
    for cat in sorted(cov):
        pre, post = cov[cat]["pre"], cov[cat]["post"]
        if post == 0:
            causa = (
                f"ningún chunk en {cat} (0 pre): el corpus no cubre esa intención"
                if pre == 0 else f"{pre} chunks entraron y todos fueron descartados"
            )
            s.append(
                f"**Cobertura:** la categoría {cat} queda vacía post-dedup ({causa}): "
                "las preguntas de esa intención no podrán responderse con este índice. "
                "Añade corpus o revisa `DEDUP_UMBRAL`."
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


def _scoring_md(scoring: dict[str, Any] | None, dedup: dict[str, Any] | None) -> list[str]:
    """Sección 6 (TSD) completa: 6.1 Scoring → 6.2 Dedup → 6.3 Cobertura (una sola pasada)."""
    if scoring is None and dedup is None:
        return ["## 6. TSD (scoring + dedup)", "", "Bloque TSD desactivado (TAG_SCORING_DEDUP=false).", ""]
    lineas = ["## 6. TSD (scoring + dedup)", ""]
    if scoring is not None:
        lineas += [
            "### 6.1 Scoring (`semantic_score`)",
            "",
            "| Métrica | Valor | Interpretación |",
            "|---|---|---|",
            f"| min | {scoring.get('semantic_score_min', 0)} | score más bajo del corpus |",
            f"| media | {scoring.get('semantic_score_media', 0)} | |",
            f"| máx | {scoring.get('semantic_score_max', 0)} | score más alto del corpus |",
            f"| centralidad media | {scoring.get('centralidad_media', 0)} | coherencia con el centroide |",
            f"| redundancia media | {scoring.get('redundancia_media', 0)} | similitud con el vecino más cercano |",
            (
                f"| chunks con score ≥ 0.6 | {scoring.get('score_buenos_n', 0)} de {scoring.get('chunks_n', 0)} "
                f"({scoring.get('score_buenos_pct', 0.0)} %) | umbral interno de esta fase: "
                "NO es precisión ni relevancia frente a consultas |"
            ),
            f"| tiempo | {formatear_duracion(scoring.get('tiempo_s'))} | fase SCORING |",
            "",
        ]
        lineas += _GLOSARIO_TSD
        lineas += [""] + _redundancia_politica_md(scoring)
    if dedup is not None:
        lineas += [
            "### 6.2 Dedup",
            "",
            "| Métrica | Valor |",
            "|---|---|",
            f"| umbral coseno (`DEDUP_UMBRAL`) | {dedup.get('umbral')} |",
            f"| chunks pre | {dedup.get('chunks_pre', 0)} |",
            f"| chunks post | {dedup.get('chunks_post', 0)} |",
            f"| descartados exactos | {dedup.get('descartados_exactos', 0)} |",
            f"| descartados semánticos | {dedup.get('descartados_semantico', 0)} |",
            f"| descartados total | {dedup.get('descartados_total', 0)} ({dedup.get('descartados_pct', 0)} %) |",
            f"| tiempo | {formatear_duracion(dedup.get('tiempo_s'))} | fase DEDUP |",
            "",
        ]
    lineas += _cobertura_md(dedup)
    return lineas


def _integridad_md(integridad: dict[str, Any] | None) -> list[str]:
    """Sección 5 — línea de integridad de los vectores finales (solo si hay datos)."""
    if not integridad or integridad.get("n", 0) == 0:
        return []
    return [
        (
            f"- **Integridad:** {integridad['n']} vectores · dim {integridad['dim']} · "
            f"NaN/Inf {integridad['nan_inf']} · ceros {integridad['cero']} · "
            f"norma L2 mín {integridad['norma_min']:.4f} / media {integridad['norma_media']:.4f} / máx {integridad['norma_max']:.4f}"
        ),
    ]


def generar_informe(datos: dict[str, Any], ruta: str | Path | None = None) -> Path:
    """Escribe el informe markdown de indexación y devuelve su ruta.

    Args:
        datos: métricas y parámetros ensamblados por el pipeline:
            rutas, parametros (lista de (variable, valor, nota)), tiempo_total_s,
            fases (tiempos por fase), resumen_fases (texto por fase), num_documentos,
            num_chunks_pre_dedup/post_dedup, chunk_stats, dim_embedding, dim_msg,
            preflight, indice (nombre/vectores/space), scoring, dedup,
            fuentes ([(fuente, {docs, chunks_pre, chunks_post}), ...]), integridad.
        ruta: ruta del informe (por defecto
            ``output/informe_index_<guid8_chroma>_aaaaMMdd_hhmm.md``, con la fecha y
            hora locales de la ejecución).

    Returns:
        Ruta absoluta del informe escrito.
    """
    if ruta is None:
        guid_chroma = (datos.get("indice") or {}).get("guid_chroma")
        sufijo = f"_{guid_chroma[:8]}" if guid_chroma else ""
        ruta = f"output/informe_index{sufijo}_{datetime.now(timezone.utc).astimezone():%Y%m%d_%H%M}.md"
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
    if not preflight.get("dim_modelo"):
        lineas.append(
            "- **Dim máxima del modelo (`EMBED_DIM_MAX_*`):** no declarada "
            "(el preflight solo verifica la disponibilidad, no la dimensión)"
        )
    lineas += [
        f"- **Aviso:** {preflight.get('aviso') or 'sin avisos'}",
        "",
        "## 3. Métricas por fase",
        "",
        _fases_md(fases, resumen_fases),
        "",
        "## 4. Corpus y chunking (longitudes en caracteres)",
        "",
        "> **fuente** = archivo original · **documento** = salida de LOAD (página/fila/grupo) · **chunk** = salida de CHUNK (íntegro si entidad `no_chunk`).",
        "",
        *_fuentes_md(datos.get("fuentes") or []),
        *([""] if (datos.get("fuentes") or []) else []),
        "| Métrica (caracteres) | Valor |",
        "|---|---|",
        f"| min | {stats.get('min', 0)} |",
        f"| p25 | {stats.get('p25', 0)} |",
        f"| media | {stats.get('media', 0)} |",
        f"| p50 | {stats.get('p50', 0)} |",
        f"| p75 | {stats.get('p75', 0)} |",
        f"| max | {stats.get('max', 0)} |",
        f"| chunks cortos (< 50) | {stats.get('cortos', 0)} |",
        f"| íntegras (`no_chunk`) | {stats.get('sin_trocear', 0)} |",
        "",
        "> Un `max` mayor que `CHUNK_SIZE` no es un error: son las entidades indexadas íntegras (una fila = un chunk).",
        "",
        "## 5. Índice final (ChromaDB)",
        "",
        f"- **Colección:** `{indice.get('nombre', '—')}`",
        f"- **Vector space:** `{indice.get('space', '—')}` (coseno)",
        f"- **Dim de embedding:** {indice.get('dim', datos.get('dim_embedding', 0))}",
        f"- **Vectores insertados:** {indice.get('vectores_insertados', datos.get('num_chunks_post_dedup', 0))}",
        f"- **Vectores totales en la colección:** {indice.get('vectores_totales', '—')}",
        f"- **Recreado (`--recreate-index`):** {indice.get('recreado', False)}",
        *_integridad_md(datos.get("integridad")),
        "",
        *_scoring_md(scoring, dedup),
        "## 7. Señales y criterios de decisión",
        "",
        * _senales(datos),
    ]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta
