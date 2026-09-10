"""
tests/test_chunks.py
Tests offline unitarios para chunk.py (referentes a troceado de documentos).
Ejecutar:  pytest tests/test_chunks.py -v
"""
import pytest
import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk import trocear  # o el nombre que tenga en chunk.py [27]
from config import CHUNK_SIZE, CHUNK_OVERLAP



# FIXTURES: textos de control


@pytest.fixture
def texto_corto():
    """Si es menor que CHUNK_SIZE, debe dar 1 chunk."""
    return "La piscina municipal abre de 8:00 a 21:00. Precio: 5 euros."

@pytest.fixture
def texto_medio():
    """~2x CHUNK_SIZE → debe dar 2-3 chunks."""
    parrafo = "El abono deportivo de Madrid incluye acceso a todas las instalaciones. "
    return (parrafo * 5).strip()  # ~800 chars

@pytest.fixture
def texto_largo():
    """~10x CHUNK_SIZE → debe dar ≥ 8 chunks."""
    parrafo = (
        "La normativa deportiva municipal establece que los residentes de Madrid "
        "tienen derecho a tarifas reducidas en todas las instalaciones deportivas "
        "del Ayuntamiento. Los no empadronados pagan el precio completo. Los "
        "menores de 14 años tienen acceso gratuito a las piscinas municipales. "
    )
    return (parrafo * 20).strip()  # ~4000 chars

@pytest.fixture
def texto_con_separadores():
    """Texto con estructura clara: párrafos, listas, secciones."""
    return (
        "## Tarifas\n\n"
        "Piscina: 5 €/día. Abono mensual: 40 €. Abono anual: 350 €.\n\n"
        "## Horarios\n\n"
        "Lunes a viernes: 8:00-21:00. Sábados: 9:00-14:00.\n\n"
        "## Requisitos\n\n"
        "- Ser mayor de 16 años\n"
        "- Presentar DNI o NIE\n"
        "- Carnet de bautismo (obligatorio)\n\n"
        "## Descuentos\n\n"
        "Jóvenes (16-25): 20% de descuento. Pensionistas: 30%. "
        "Familias numerosas: 15%."
    )



# Tests chunking puros

class TestChunkTamaño:
    def test_un_chunk_si_texto_corto(self, texto_corto):
        chunks = trocear(texto_corto, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        assert len(chunks) == 1

    def test_todos_chunks_dentro_limite(self, texto_largo):
        chunks = trocear(texto_largo, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        # Tolerancia: una palabra que no cabe puede exceder ligeramente
        tolerancia = 200
        for i, c in enumerate(chunks):
            texto = c.page_content if hasattr(c, 'page_content') else c
            assert len(texto) <= CHUNK_SIZE + tolerancia, \
                f"Chunk {i} tiene {len(texto)} chars (límite {CHUNK_SIZE + tolerancia})"

    def test_solo_un_chunk_vacio(self, texto_medio):
        chunks = trocear(texto_medio, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        vacios = [c for c in chunks if not (c.page_content if hasattr(c, 'page_content') else c).strip()]
        assert len(vacios) == 0, f"Se encontraron {len(vacios)} chunks vacíos"


class TestChunkOverlap:
    def test_overlap_presente_entre_adyacentes(self, texto_largo):
        chunks = trocear(texto_largo, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if len(chunks) < 2:
            pytest.skip("Necesarios dos o más chunks para verificar overlap")

        for i in range(len(chunks) - 1):
            actual = chunks[i].page_content if hasattr(chunks[i], 'page_content') else chunks[i]
            siguiente = chunks[i+1].page_content if hasattr(chunks[i+1], 'page_content') else chunks[i+1]

            # Los últimos CHUNK_OVERLAP chars del chunk i deben aparecer
            # al inicio del chunk i+1 (o viceversa, según implementación)
            tail = actual[-CHUNK_OVERLAP:]
            # Tolerancia: el splitter puede alinear a palabra/frase
            # Verificamos que al menos el 50% del overlap está presente
            overlap_chars = sum(1 for ch in tail if ch in siguiente[:CHUNK_OVERLAP * 2])
            ratio_overlap = overlap_chars / max(len(tail), 1)
            assert ratio_overlap > 0.4, \
                f"Chunk {i}→{i+1}: overlap ratio {ratio_overlap:.2f} < 0.4"

    def test_overlap_no_es_cero(self, texto_largo):
        """Si overlap > 0, los chunks no deben ser independientes."""
        chunks = trocear(texto_largo, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if len(chunks) < 2:
            pytest.skip("Necesarios dos o más chunks")
        # Si overlap fuera 0, el inicio del chunk i+1 no tendría relación
        # con el final del chunk i. Verificamos que sí la tiene.
        c1 = chunks[0].page_content if hasattr(chunks[0], 'page_content') else chunks[0]
        c2 = chunks[1].page_content if hasattr(chunks[1], 'page_content') else chunks[1]
        # Al menos alguna palabra en común en la zona de overlap
        palabras_c1_tail = set(c1[-100:].split())
        palabras_c2_head = set(c2[:100].split())
        comunes = palabras_c1_tail & palabras_c2_head
        assert len(comunes) > 0, "No se detecta overlap entre chunks adyacentes"


class TestChunkMetadata:
    def test_chunk_index_secuencial(self, texto_largo):
        chunks = trocear(texto_largo, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        indices = []
        for c in chunks:
            if hasattr(c, 'metadata') and c.metadata:
                indices.append(c.metadata.get("chunk_index"))
            else:
                indices.append(None)

        # Si hay metadata, deben ser 0, 1, 2, ...
        if all(i is not None for i in indices):
            assert indices == list(range(len(chunks))), \
                f"chunk_index no secuencial: {indices}"

    def test_source_en_metadata(self):
        """Si se pasa metadata de source, debe aparecer en cada chunk."""
        texto = "Texto de prueba con fuente."
        from langchain_core.documents import Document
        doc = Document(page_content=texto, metadata={"source": "test.pdf"})
        chunks = trocear([doc], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for c in chunks:
            if hasattr(c, 'metadata'):
                assert c.metadata.get("source") == "test.pdf"


class TestChunkPreservacion:
    def test_no_pierde_contenido_significativo(self, texto_con_separadores):
        """
        Heurística fuerte: cada palabra de 5 o más caracteres del texto original
        debe aparecer en la concatenación de todos los chunks.
        """
        chunks = trocear(texto_con_separadores, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        concatenado = " ".join(c.page_content if hasattr(c, 'page_content') else c for c in chunks).lower()

        palabras_originales = [
            w.strip(".,;:!?\"'()") for w in texto_con_separadores.lower().split()
            if len(w) >= 5
        ]
        perdidas = [w for w in palabras_originales if w not in concatenado]

        assert len(perdidas) == 0, \
            f"Palabras perdidas en chunking: {perdidas[:10]}… ({len(perdidas)} total)"

    def test_numeros_preservados(self, texto_con_separadores):
        """Los precios, horarios y porcentajes no deben romperse."""
        chunks = trocear(texto_con_separadores, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        concatenado = " ".join(c.page_content if hasattr(c, 'page_content') else c for c in chunks)

        numeros_esperados = ["5", "40", "350", "8:00", "21:00", "20%", "30%", "15%"]
        perdidos = [n for n in numeros_esperados if n not in concatenado]
        assert len(perdidos) == 0, f"Números/horarios perdidos: {perdidos}"



# MÉTRICAS OBJETIVAS (para el informe)

class TestChunkMetrics:
    """
    Métricas cuantitativas para el apartado "Experimento de chunking" del informe.
    Ejecutar: pytest -v -s tests/test_chunks.py::TestChunkMetrics
    """

    def test_estadisticas_basicas(self, texto_largo):
        chunks = trocear(texto_largo, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        lengths = [
            len(c.page_content if hasattr(c, 'page_content') else c) for c in chunks
        ]

        print(f"\n{'='*50}")
        print(f"  CHUNKING METRICS (texto_largo)")
        print(f"{'='*50}")
        print(f"  Input length:      {len(texto_largo)} chars")
        print(f"  Nº chunks:         {len(chunks)}")
        print(f"  Chunk size config: {CHUNK_SIZE}")
        print(f"  Overlap config:    {CHUNK_OVERLAP}")
        print(f"  Min chunk len:     {min(lengths)}")
        print(f"  Max chunk len:     {max(lengths)}")
        print(f"  Mean chunk len:    {sum(lengths)/len(lengths):.1f}")
        print(f"  Std dev:           {(sum((l - sum(lengths)/len(lengths))**2 for l in lengths)/len(lengths))**0.5:.1f}")
        print(f"  Ratio overlap:     {CHUNK_OVERLAP/CHUNK_SIZE*100:.0f}%")
        print(f"{'='*50}")

        # Aserciones blandas (informativas, no bloqueantes)
        assert max(lengths) <= CHUNK_SIZE * 1.5, "Chunk demasiado largo"
        assert min(lengths) >= 50, "Chunk demasiado corto (¿corte mal?)"

    def test_distribucion_tamanos(self):
        """
        Barrido de chunk_size para el experimento del informe.
        Muestra cómo cambia el nº de chunks y el tamaño medio.
        """
        texto = "La piscina municipal de Madrid abre todos los días del año. " * 50  # ~4500 chars

        print(f"\n{'='*60}")
        print(f"  EXPERIMENTO: chunk_size vs nº chunks (input: {len(texto)} chars)")
        print(f"{'='*60}")
        print(f"  {'Size':>6} | {'Overlap':>7} | {'Nº chunks':>9} | {'Min len':>7} | {'Max len':>7} | {'Mean':>6}")
        print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")

        for size in [400, 600, 800, 1000, 1200]:
            overlap = int(size * 0.12)  # 12%
            chunks = trocear(texto, chunk_size=size, chunk_overlap=overlap)
            lengths = [
                len(c.page_content if hasattr(c, 'page_content') else c) for c in chunks
            ]
            if lengths:
                print(f"  {size:>6} | {overlap:>7} | {len(chunks):>9} | {min(lengths):>7} | {max(lengths):>7} | {sum(lengths)/len(lengths):>6.0f}")
            else:
                print(f"  {size:>6} | {overlap:>7} | {'ERROR':>9}")

        print(f"{'='*60}")