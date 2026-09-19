from __future__ import annotations

import json
import csv
import os
import getpass
from datetime import datetime

if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Pega aquí tu GEMINI_API_KEY: ")

from src.logic import responder

# Cargar preguntas
with open("queries/evaluation_questions.json", "r", encoding="utf-8") as f:
    preguntas = json.load(f)

TOP_KS = [1, 3, 5]
resultados = []

for top_k in TOP_KS:
    print(f"\n{'='*50}")
    print(f"TOP_K = {top_k}")
    print(f"{'='*50}")

    for p in preguntas:
        print(f"  [{p['id']}] {p['pregunta'][:60]}...")
        resultado = responder(p["pregunta"], top_k=top_k)

        metrics = resultado.get("metrics", {})
        resultados.append({
            "id": p["id"],
            "categoria": p["categoria_esperada"],
            "out_of_corpus": p.get("out_of_corpus", False),
            "pregunta": p["pregunta"],
            "top_k": top_k,
            "abstained": resultado.get("abstained", False),
            "error": resultado.get("error", ""),
            "respuesta": resultado.get("respuesta", "")[:300],
            "fuentes": ", ".join(resultado.get("fuentes", [])),
            "n_chunks": metrics.get("n_chunks", ""),
            "modelo": metrics.get("model", ""),
            "t_retrieval": metrics.get("retrieval", ""),
            "t_generation": metrics.get("generation", ""),
        })

        print(f"  → abstained={resultado.get('abstained')} | chunks={metrics.get('n_chunks')} | t_gen={metrics.get('generation')}")

# Guardar CSV
os.makedirs("output", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ruta = f"output/eval_{timestamp}.csv"

with open(ruta, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
    writer.writeheader()
    writer.writerows(resultados)

print(f"\nResultados guardados en {ruta}")