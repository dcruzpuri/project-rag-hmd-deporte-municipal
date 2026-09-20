import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieve import recuperar

for i, c in enumerate(recuperar('¿Cuánto cuesta una instalación deportiva?', top_k=3), 1):
    dist = c["distance"]
    src = c["source"]
    texto = c["text"][:300]
    print(f"--- {i} | dist={dist:.4f} | source={src} ---")
    print(texto)
    print()