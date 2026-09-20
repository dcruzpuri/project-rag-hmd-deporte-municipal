# tests/test_retrieve_varias.py — versión mejorada
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieve import recuperar

preguntas = [
    "¿Qué piscinas municipales hay en Madrid?",
    "¿Cuál es el horario de la piscina?",
    "¿Qué instalaciones deportivas hay en Chamberí?",
    "¿Qué descuentos hay para abonados?",
    "¿Cuál es la capital de Francia?",
]

resultados = []
for k in [1, 3, 5]:
    print(f"\n########## TOP_K = {k} ##########")
    for p in preguntas:
        print(f"\n=== {p} ===")
        chunks = recuperar(p, top_k=k)
        for i, c in enumerate(chunks, 1):
            print(f"  {i} | dist={c['distance']:.4f} | {c['source']}")
        resultados.append({
            "top_k": k,
            "pregunta": p,
            "chunks": [{"dist": c["distance"], "source": c["source"]} for c in chunks],
        })

Path("output/retrieval_eval.json").write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print("\nGuardado en output/retrieval_eval.json")