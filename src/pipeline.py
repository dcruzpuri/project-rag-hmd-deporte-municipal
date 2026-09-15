"""
src/pipeline.py
Encadena PREFLIGHT → LOAD → CLEAN → TAG → CHUNK → EMBED → SCORING → DEDUP → INDEX.
PREFLIGHT garantiza que EMBED_MODEL esté disponible en EMBED_PROVIDER antes de empezar.
Es el punto de entrada que importa tu aplicación [9].
Al terminar genera un informe markdown
(output/informe_index_<guid8_chroma>_aaaaMMdd_hhmm.md) con los
parámetros aplicados y las métricas de la ejecución para la toma de decisiones.
"""
import statistics
import time
from typing import Any

import config
from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    COSINE_SPACE,
    DEDUP_UMBRAL,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    EMBED_PROVIDER,
    EXPORT_EMBEDDINGS,
    GEN_MODEL,
    GEN_PROVIDER,
    HF_DEVICE,
    OLLAMA_BASE_URL,
    TAG_MODEL,
    TAG_PROVIDER,
    TAG_SCORING_DEDUP,
)

from .chunk import trocear
from .clean import limpiar
from .embed import embeddear, exportar_json, verificar_modelo_disponible
from .index import indexar, obtener_guid_chroma
from .informe import generar_informe
from .load import cargar_archivos
from .tsd.dedup import deduplicar
from .tsd.scoring import puntuar
from .tsd.tag import etiquetar


def _log(fase: str, msg: str) -> None:
    """Marca de tiempo + fase: formato único en consola (evita repetir strftime)."""
    print(time.strftime("%Y-%m-%d %H:%M:%S") + f" [{fase}] {msg}")


def _comprobar_dim(dim: int, embed_dim: int,
                   dim_max_modelo: int | None) -> tuple[str, str]:
    """
    Compara la dim final del índice con EMBED_DIM declarado en .env.

    ``dim`` ya es la dim del índice (embeddear() recorta a EMBED_DIM si el
    modelo produce más), así que siempre dim <= embed_dim.

    Args:
        dim: dim final de los vectores insertados (o 0 si corpus vacío).
        embed_dim: EMBED_DIM de .env (recorte declarado).
        dim_max_modelo: dim máxima declarada en el preflight
            (EMBED_DIM_MAX_<proveedor>); None = no declarada.

    Returns:
        (nivel, mensaje) donde nivel ∈ {"vacio", "ok", "info", "aviso"}:
        - "vacio": no hay embeddings (corpus vacío).
        - "ok":    dim == EMBED_DIM y no hay dim máxima declarada mayor.
        - "info":  EMBED_DIM < dim máxima del modelo → el índice se crea con
            EMBED_DIM (se informa de la dim máxima que maneja el modelo).
        - "aviso": EMBED_DIM > dim del modelo → EMBED_DIM se ajusta al máximo del modelo
    """
    if not dim:
        return "vacio", "no se generaron embeddings (corpus vacío)"
    if embed_dim > dim:
        return "aviso", (
            f"EMBED_DIM ({embed_dim}) supera la dim máxima del modelo ({dim}): "
            f"EMBED_DIM ajustado a {dim} — las dimensiones con las que se crea "
            f"el índice son el máximo que maneja el modelo"
        )
    if dim_max_modelo is not None and dim_max_modelo > dim:
        return "info", (
            f"EMBED_DIM ({embed_dim}) es menor que la dim máxima del modelo "
            f"({dim_max_modelo}): el modelo maneja hasta {dim_max_modelo} dims; "
            f"el índice se crea con {embed_dim} dims (EMBED_DIM)"
        )
    return "ok", ""


def _stats_chunks(chunks) -> dict[str, float]:
    """Distribución de longitudes de chunk (métricas de apuntes de RAG Engineering)."""
    longitudes = [len(c.page_content) for c in chunks]
    if not longitudes:
        return {"min": 0, "p25": 0, "media": 0, "p50": 0, "p75": 0, "max": 0, "cortos": 0}
    if len(longitudes) >= 4:
        p25, p50, p75 = statistics.quantiles(longitudes, n=4)
    else:
        p25 = p50 = p75 = statistics.median(longitudes)
    return {
        "min": min(longitudes),
        "p25": p25,
        "media": round(statistics.mean(longitudes), 1),
        "p50": p50,
        "p75": p75,
        "max": max(longitudes),
        "cortos": sum(1 for l in longitudes if l < 50),  # chunks demasiado cortos (ruido)
    }


def ejecutar_pipeline(
    rutas: list[str] | str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    persist_dir: str | None = None,
    collection_name: str | None = None,
    recreate_index: bool = False,
    export_embeddings: bool = False,
    informe: bool = True,
) -> dict[str, Any]:
    """
    Ejecuta el pipeline completo de RAG (con Tagging-Scoring-Deduplication opcional).

    Returns:
        dict con métricas (documentos, chunks, stats, dedup, tiempo) y las claves
        que consume el informe: ``parametros``, ``fases``, ``resumen_fases``,
        ``scoring``, ``dedup``, ``preflight``, ``indice``. La generación del informe
        se puede saltar con ``informe=False`` (por ejemplo en los tests).
    Uso:
        from rag_pipeline import ejecutar_pipeline
        ejecutar_pipeline("./docs/corpus/")
    """
    rutas_norm = [str(r) for r in (rutas if isinstance(rutas, list) else [rutas])]
    t_inicio = time.perf_counter()
    fases: dict[str, float] = {}

    def _crono(fase: str, t0: float) -> None:
        """Registra el tiempo (s) de una fase terminada (precisión real, sin redondeo)."""
        fases[fase] = time.perf_counter() - t0

    # --- PREFLIGHT: comprobar que el modelo existe en el proveedor ANTES de empezar
    t_p0 = time.perf_counter()
    preflight = verificar_modelo_disponible()
    _crono("PREFLIGHT", t_p0)
    if preflight["aviso"]:
        _log("PREFLIGHT", f"info: {preflight['aviso']}")
    _log("PREFLIGHT", f"modelo {EMBED_MODEL} disponible en {EMBED_PROVIDER}")

    # --- LOAD
    t_load0 = time.perf_counter()
    _log("LOAD", "inicio")
    documentos = cargar_archivos(rutas)
    n_cargados = len(documentos)
    _log("LOAD", f"{n_cargados} documentos cargados")
    _crono("LOAD", t_load0)

    # --- CLEAN
    t_clean0 = time.perf_counter()
    _log("CLEAN", "inicio")
    documentos = limpiar(documentos)
    n_normalizados = len(documentos)
    _log("CLEAN", f"{n_normalizados} documentos normalizados")
    _crono("CLEAN", t_clean0)

    # --- TAG (por fuente, antes de trocear)
    tag_info: dict[str, Any] = {}
    if TAG_SCORING_DEDUP:
        t_tag = time.perf_counter()
        _log("TAG", "inicio (LLM)")
        documentos = etiquetar(documentos)
        n_fuentes = len({d.metadata.get("source", "") for d in documentos})
        tag_info = {"tiempo_s": round(time.perf_counter() - t_tag, 3), "fuentes": n_fuentes}
        _crono("TAG", t_tag)
        _log("TAG", f"{n_fuentes} fuentes etiquetadas")

    # --- CHUNK → hereda doc_category, tags, relevancia_llm del documento padre [7]
    _log("CHUNK", "inicio")
    t_chunk0 = time.perf_counter()
    chunks = trocear(documentos, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    stats = _stats_chunks(chunks)
    _log("CHUNK", (f"{len(chunks)} chunks | min {stats['min']} · p25 {stats['p25']} · "
                   f"media {stats['media']} · p75 {stats['p75']} · max {stats['max']} · "
                   f"cortos(<50) {stats['cortos']}"))
    _crono("CHUNK", t_chunk0)

    # --- EMBED
    _log("EMBED", "inicio")
    t_embed0 = time.perf_counter()
    # SEGURIDAD: dim final = min(dim del modelo, EMBED_DIM). embeddear() es la
    # puerta única: recorta a EMBED_DIM (corte + renormalización) si el modelo
    # produce más, así que la dim resultante nunca supera jamás al modelo ni a EMBED_DIM.
    embeddings = embeddear([c.page_content for c in chunks])
    dim = len(embeddings[0]) if embeddings else 0
    _log("EMBED", f"{len(embeddings)} vectores de {dim} dims "
           f"(EMBED_DIM declarado: {config.EMBED_DIM})")

    dim_max = preflight.get("dim_modelo")
    nivel_dim, msg_dim = _comprobar_dim(dim, config.EMBED_DIM, dim_max)
    if nivel_dim == "vacio":
        _log("EMBED", f"AVISO: {msg_dim}")
    elif nivel_dim == "info":
        _log("EMBED", f"INFO: {msg_dim}")
    elif nivel_dim == "aviso":
        # Ajustamos el valor en memoria: las dimensiones con las que se crea el
        # índice son el máximo del modelo (dim), nunca un EMBED_DIM mayor.
        config.EMBED_DIM = dim
        _log("EMBED", f"AVISO: {msg_dim}")
    _crono("EMBED", t_embed0)

    # --- SCORING + DEDUP (requiere tag previo: relevancia_llm en metadata)
    n_pre_dedup = len(chunks)
    tsd: dict[str, Any] = {}
    t_tsd0 = time.perf_counter()
    if TAG_SCORING_DEDUP:
        _log("TSD", "scoring + dedup")
        chunks = puntuar(chunks, embeddings, info=tsd)
        _log("DEDUP", "deduplicación")
        chunks, embeddings = deduplicar(chunks, embeddings, info=tsd)
        _crono("TSD", t_tsd0)

    # --- INDEX
    _log("INDEX", "inicio")
    t_index0 = time.perf_counter()
    # id global estable y único: fuente + posición secuencial global
    # (chunk_index es secuencial POR DOCUMENTO, y un CSV produce muchos
    # documentos con la misma fuente, lo que haría ids duplicados y
    # Chroma sobreescribiría los chunks anteriores)
    ids = [f"{c.metadata.get('source', 'doc')}::{i}"
           for i, c in enumerate(chunks)]
    vivos, totales = indexar(
        ids=ids,
        embeddings=embeddings,
        documents=[c.page_content for c in chunks],
        metadatos=[c.metadata for c in chunks],
        persist_dir=persist_dir,
        collection_name=collection_name,
        recreate=recreate_index,
        )
    guid_chroma = obtener_guid_chroma(persist_dir)

    if export_embeddings or EXPORT_EMBEDDINGS:
        exportar_json(chunks, embeddings)
    _crono("INDEX", t_index0)

    tiempo_total = time.perf_counter() - t_inicio
    _crono("FIN", t_inicio)
    tiempo_total_txt = (f"{tiempo_total * 1000:.1f} ms" if tiempo_total < 1
                        else f"{tiempo_total:.2f} s")
    _log("FIN", f"pipeline completado en {tiempo_total_txt}")
    _log("INFORME", "generando informe de indexación (markdown)")

    # Tabla «Parámetros aplicados»: variables siempre relevantes más filas
    # condicionales (solo aquellas que han influido en esta ejecución). Nunca se
    # vuelcan credenciales (GOOGLE_API_KEY, HF_TOKEN): sus valores viven en .env.
    # EMBED_DIM_MAX_* se omite aquí: solo informa cuando el preflight cayó en
    # caché (.env) y queda recogida en la sección 2 (Preflight).
    parametros: list[tuple[str, Any, str]] = [
        ("EMBED_PROVIDER", EMBED_PROVIDER, "proveedor de embeddings"),
        ("EMBED_MODEL", EMBED_MODEL, "modelo de embeddings"),
        ("EMBED_DIM", config.EMBED_DIM, " dimensión máxima (min(dim modelo, EMBED_DIM))"),
        ("EMBED_BATCH_SIZE", EMBED_BATCH_SIZE, "tamaño de lote de embeddings"),
        ("TAG_PROVIDER", TAG_PROVIDER, "proveedor del etiquetado LLM"),
        ("TAG_MODEL", TAG_MODEL, "modelo del etiquetado LLM"),
        ("GEN_PROVIDER", GEN_PROVIDER, "proveedor de generación"),
        ("GEN_MODEL", GEN_MODEL, "modelo de generación"),
    ]
    if "ollama" in (EMBED_PROVIDER, TAG_PROVIDER, GEN_PROVIDER):
        parametros.append(("OLLAMA_BASE_URL", OLLAMA_BASE_URL, "endpoint de Ollama"))
    if "huggingface" in (EMBED_PROVIDER, TAG_PROVIDER, GEN_PROVIDER):
        parametros.append(("HF_DEVICE", HF_DEVICE, "dispositivo para modelos HuggingFace"))
    parametros += [
        ("CHUNK_SIZE", chunk_size if chunk_size is not None else CHUNK_SIZE,
         "longitud de chunk"),
        ("CHUNK_OVERLAP", chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP,
         "sobrelap de chunks"),
        ("TAG_SCORING_DEDUP", str(TAG_SCORING_DEDUP), "bloque TSD activo"),
        ("DEDUP_UMBRAL", DEDUP_UMBRAL, "umbral coseno de dedup"),
        ("CHROMA_DIR", persist_dir or CHROMA_DIR, "directorio de ChromaDB"),
        ("COLLECTION_NAME", collection_name or COLLECTION_NAME, "colección de vectores"),
        ("COSINE_SPACE", COSINE_SPACE, "métrica de similitud"),
        ("recreate_index", str(recreate_index), "la colección se borró antes de indexar"),
        ("EXPORT_EMBEDDINGS", str(export_embeddings or EXPORT_EMBEDDINGS),
         "exportación a output/embeddings.json"),
    ]

    info: dict[str, Any] = {
        "rutas": rutas_norm,
        "parametros": parametros,
        "preflight": {
            "verificado_por": preflight.get("verificado_por", "online"),
            "dim_modelo": dim_max,
            "aviso": preflight.get("aviso"),
        },
        "fases": fases,
        "resumen_fases": {
            "PREFLIGHT": "ok (modelo disponible)",
            "LOAD": f"{n_cargados} documentos cargados",
            "CLEAN": f"{n_normalizados} documentos normalizados",
            **(
                {"TAG": f"{tag_info['fuentes']} fuentes etiquetadas (LLM)"}
                if TAG_SCORING_DEDUP
                else {"TAG": "desactivado (TAG_SCORING_DEDUP=false)"}
            ),
            "CHUNK": f"{n_pre_dedup} chunks generados (min {stats['min']}, media {stats['media']})",
            "EMBED": f"{len(embeddings)} vectores de {dim} dims",
            **(
                {"TSD": f"scoring + dedup: {n_pre_dedup - len(chunks)} chunks descartados"}
                if TAG_SCORING_DEDUP
                else {}
            ),
            "INDEX": f"{vivos} vectores insertados de {totales} totales",
            "FIN": f"pipeline completado en {tiempo_total_txt}",
        },
        "num_documentos_cargados": n_cargados,
        "num_documentos_normalizados": n_normalizados,
        "n_fuentes": tag_info.get("fuentes") if TAG_SCORING_DEDUP else None,
        "num_documentos": n_normalizados,
        "num_chunks_pre_dedup": n_pre_dedup,
        "num_chunks_post_dedup": len(chunks),
        "chunks_descartados": n_pre_dedup - len(chunks),
        "dim_embedding": dim,
        "dim_msg": msg_dim,
        "chunk_stats": stats,
        "scoring": tsd.get("scoring"),
        "dedup": tsd.get("dedup"),
        "indice": {
            "nombre": collection_name or COLLECTION_NAME,
            "guid_chroma": guid_chroma,
            "space": COSINE_SPACE,
            "dim": dim,
            "vectores_insertados": vivos,
            "vectores_totales": totales,
            "recreado": recreate_index,
        },
        "tiempo_total_s": tiempo_total,
    }

    if informe:
        ruta_informe = generar_informe(info)
        _log("INFORME", f"informe generado: {ruta_informe}")

    return info


if __name__ == "__main__":
    # Uso (desde la raíz del proyecto):
    #   python -m src.pipeline data --recreate-index
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline RAG completo: LOAD → CLEAN → TAG → CHUNK → EMBED → SCORING → DEDUP → INDEX"
    )
    parser.add_argument("rutas", nargs="+", help="Archivos o carpetas del corpus (p. ej. data)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Longitud de chunk (por defecto: CHUNK_SIZE de .env)")
    parser.add_argument("--chunk-overlap", type=int, default=None,
                        help="Sobrelap de chunks (por defecto: CHUNK_OVERLAP de .env)")
    parser.add_argument("--persist-dir", default=None,
                        help="Directorio de ChromaDB (por defecto: CHROMA_DIR de .env)")
    parser.add_argument("--collection", default=None,
                        help="Nombre de la colección (por defecto: COLLECTION_NAME de .env)")
    parser.add_argument("--recreate-index", action="store_true",
                        help="Borrar la colección antes de indexar")
    args = parser.parse_args()

    print(ejecutar_pipeline(
        args.rutas,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        recreate_index=args.recreate_index,
    ))
