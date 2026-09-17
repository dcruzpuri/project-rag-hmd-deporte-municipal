import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.logic import responder, rag_ask


def probar(pregunta: str):
    print(f"\n{'='*80}")
    print(f"PREGUNTA: {pregunta}")
    print('='*80)
    resultado = responder(pregunta, top_k=5)
    print(f"abstained: {resultado['abstained']}")
    print(f"error:     {resultado['error']}")
    print(f"fuentes:   {resultado['fuentes']}")
    print(f"metrics:   {resultado['metrics']}")
    print(f"\nRESPUESTA:\n{resultado['respuesta']}")


if __name__ == "__main__":
    # 1) In-corpus, con respuesta esperada
    probar("¿Qué piscinas municipales hay en Madrid?")
    # 2) In-corpus, con datos concretos
    probar("¿Qué instalaciones deportivas hay en Chamberí?")
    # 3) Fuera de corpus -> debe abstenerse
    probar("¿Cuál es la capital de Francia?")
    # 4) rag_ask (contrato para Agentes)
    print(f"\n{'='*80}")
    print("PRUEBA rag_ask (solo string):")
    print('='*80)
    print(rag_ask("¿Qué descuentos hay para abonados?"))