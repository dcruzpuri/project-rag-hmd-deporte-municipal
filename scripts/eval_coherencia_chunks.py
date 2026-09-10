"""
scripts/eval_coherencia_chunks.py
Métrica opcional: evalúa si los chunks son cortados a la mitad.
Usa los embeddings de Ollama. Ejecutar UNA ÚNICA VEZ y volcar 
los datos obtenidos al informe.
Ejecución: python -m scripts.eval_coherencia_chunks
"""
import sys
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL
from src.chunk import trocear
from src.embed import embeddear  # ya existe
from src.clean import limpiar

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def main():
    # Texto de ejemplo del dominio
    texto = (
        "La normativa deportiva municipal establece que los residentes de Madrid "
        "tienen derecho a tarifas reducidas en todas las instalaciones deportivas. "
        "Los no empadronados pagan el precio completo. Los menores de 14 años "
        "tienen acceso gratuito a las piscinas municipales durante el verano. "
        "El abono anual cuesta 350 euros para residentes y 450 para no residentes. "
        "La piscina abre de 8:00 a 21:00 en semana y de 9:00 a 14:00 en sábado. "
    ) * 5

    chunks = trocear(texto, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    textos = [c.page_content if hasattr(c, 'page_content') else c for c in chunks]

    print(f"Modelo: {EMBED_MODEL}")
    print(f"Chunks: {len(textos)}")
    print("Embedding…")
    vectors = embeddear(textos)

    # Similitud entre adyacentes vs. pares aleatorios
    adyacentes = []
    aleatorios = []

    for i in range(len(textos) - 1):
        adyacentes.append(cosine_sim(vectors[i], vectors[i + 1]))

    for i, j in combinations(range(len(vectors)), 2):
        if j != i + 1:  # excluir adyacentes
            aleatorios.append(cosine_sim(vectors[i], vectors[j]))

    mean_ad = sum(adyacentes) / len(adyacentes) if adyacentes else 0
    mean_al = sum(aleatorios) / len(aleatorios) if aleatorios else 0
    margen = mean_ad - mean_al

    print(f"\n{'='*50}")
    print(f"  COHERENCIA SEMÁNTICA DE CHUNKS")
    print(f"{'='*50}")
    print(f"  Sim. media adyacentes:  {mean_ad:.4f}")
    print(f"  Sim. media aleatorios:  {mean_al:.4f}")
    print(f"  Margen (ady − aleat):   {margen:+.4f}")
    print(f"{'='*50}")
    print(f"  Interpretación: un margen > 0.05 indica que el overlap")
    print(f"  mantiene coherencia temática entre chunks adyacentes.")
    print(f"  Un margen similar a 0 sugiere cortes arbitrarios (semántica comprometida o partida).")

if __name__ == "__main__":
    main()