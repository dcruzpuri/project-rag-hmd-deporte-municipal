"""
Test rápido de modelos de embedding en Ollama.
4 frases: tres relacionadas, una irrelevante.
Calcula la similitud coseno entre cada par y muestra la tabla.

Uso:
    python scripts/test_embeddings.py
"""

import requests
import numpy as np

OLLAMA_URL = "http://127.0.0.1:11434/api/embed"

#  Frases de prueba
FRASES = [
    "¿Cuánto cuesta el abono mensual de la piscina municipal?",     # A
    "Precio de la tarifa de uso de la piscina para residentes",      # B (sinónimo)
    "Tarifa del abono deportivo de la piscina municipal",            # C (paráfrasis)
    "Receta de tortilla española con patatas",                       # D (irrelevante)
]

#  Modelos Ollama de embedding a probar
MODELOS = ["nomic-embed-text", "nomic-embed-text-v2-moe", 
           "mxbai-embed-large:335M", "locusai/all-minilm-l6-v2"]


def embedder(model: str, textos: list[str]) -> list[list[float]]:
    """Embed batch vía Ollama."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "input": textos},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def similitud_coseno(a: list[float], b: list[float]) -> float:
    """sim = cos(). donde cuanto más se aproxima a 1.0 = mismo significado. """
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def ejecutar_test(model: str) -> None:
    print(f"\n{'='*50}")
    print(f"  Modelo: {model}")
    print(f"{'='*50}")

    vectores = embedder(model, FRASES)

    # Pares: (A,B), (A,C), (B,C)
    pares = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    etiquetas = ["A", "B", "C", "D"]

    print(f"\n  {'Par':<6} {'Relación':<18} {'Similitud':>10}")
    print(f"  {'-'*40}")

    for i, j in pares:
        sim = similitud_coseno(vectores[i], vectores[j])
        par = f"{etiquetas[i]}–{etiquetas[j]}"
        relacion = "relacionadas" if (i, j) == (0, 1) else "irrelevantes"
        print(f"  {par:<6} {relacion:<18} {sim:>10.4f}")

    #  Veredicto: A-B/A-C son paráfrasis (relevantes), D es la irrelevante
    sim_relevante  = similitud_coseno(vectores[0], vectores[1])
    sim_irrelevante = similitud_coseno(vectores[0], vectores[3])
    margen = sim_relevante - sim_irrelevante

    print(f"\n  Margen (relevante − irrelevante): {margen:.4f}")
    if margen > 0.1:
        print("  [✓] Buen separador para este dominio.")
    else:
        print("  [◬] Separacion debil; el modelo puede no distinguir bien.")
    print()


if __name__ == "__main__":
    for modelo in MODELOS:
        try:
            ejecutar_test(modelo)
        except Exception as e:
            print(f"\n  [◬] Error con {modelo}: {e}\n")
            print("    -> Seria conveniente comprobar si el modelo esta descargado y el servidor Ollama esta funcionando/respondiendo a las peticiones.\n")