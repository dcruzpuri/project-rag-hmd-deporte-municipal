"""
tests/test_clean.py
Tests unitarios offline para clean.py (referentes a normalización de texto).
Ejecutar:  pytest tests/test_clean.py -v
"""
import pytest
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clean import limpiar  



# FIXTURES: textos de prueba (no dependen de corpus real)

@pytest.fixture
def texto_con_bom():
    return "\ufeffPrecio de la piscina: 5 €"

@pytest.fixture
def texto_espacios_dobles():
    return "La  piscina   está  abierta  de  8 a 21 h"

@pytest.fixture
def texto_lineas_vacias():
    return "Línea 1\n\n\n\nLínea 2\n\n\nLínea 3"

@pytest.fixture
def texto_control_chars():
    return "Inicio\x00\x01\x02Fin"

@pytest.fixture
def texto_ya_limpio():
    return "Texto limpio, sin ruido, con un solo espacio."

@pytest.fixture
def texto_vacio():
    return ""

@pytest.fixture
def texto_solo_espacios():
    return "   \n\t  \n  "


# Test generales

class TestCleanBOM:
    def test_elimina_bom(self, texto_con_bom):
        resultado = limpiar(texto_con_bom)
        assert "\ufeff" not in resultado

    def test_preserva_contenido_despues_bom(self, texto_con_bom):
        resultado = limpiar(texto_con_bom)
        assert "Precio de la piscina" in resultado


class TestCleanEspacios:
    def test_dobles_espacios_a_uno(self, texto_espacios_dobles):
        resultado = limpiar(texto_espacios_dobles)
        assert "  " not in resultado  # ningún doble espacio

    def test_tabulaciones_normalizadas(self):
        texto = "Columna\t1\t2\t3"
        resultado = limpiar(texto)
        # Sin tabulaciones (o espacios múltiples)
        assert "\t\t" not in resultado


class TestCleanLineas:
    def test_lineas_vacias_consecutivas(self, texto_lineas_vacias):
        resultado = limpiar(texto_lineas_vacias)
        assert "\n\n\n" not in resultado  # máx 1 salto vacío

    def test_contenido_preservado(self, texto_lineas_vacias):
        resultado = limpiar(texto_lineas_vacias)
        assert "Línea 1" in resultado
        assert "Línea 2" in resultado
        assert "Línea 3" in resultado


class TestCleanControlChars:
    def test_elimina_control_chars(self, texto_control_chars):
        resultado = limpiar(texto_control_chars)
        control = re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', resultado)
        assert len(control) == 0

    def test_preserva_texto_visible(self, texto_control_chars):
        resultado = limpiar(texto_control_chars)
        assert "Inicio" in resultado
        assert "Fin" in resultado


class TestCleanEdgeCases:
    def test_texto_vacio(self, texto_vacio):
        resultado = limpiar(texto_vacio)
        assert resultado == "" or resultado is None or resultado.strip() == ""

    def test_solo_espacios(self, texto_solo_espacios):
        resultado = limpiar(texto_solo_espacios)
        assert resultado.strip() == ""

    def test_ya_limpio_no_cambia(self, texto_ya_limpio):
        resultado = limpiar(texto_ya_limpio)
        assert "Texto limpio" in resultado
        assert "ruido" in resultado



# Métricas numéricas (objetivo por eso)

class TestCleanMetrics:
    """
    Son métricas que se imprimen para el informe. 
    Ejecutar con: pytest -v -s tests/test_clean.py::TestCleanMetrics
    """

    def test_ratio_compresion(self, texto_con_bom, texto_espacios_dobles, texto_lineas_vacias):
        textos = [texto_con_bom, texto_espacios_dobles, texto_lineas_vacias]
        ratios = []
        for t in textos:
            limpiado = limpiar(t)
            if len(t) > 0:
                ratios.append(len(limpiado) / len(t))
        ratio_prom = sum(ratios) / len(ratios)
        print(f"\n[ MÉTRICA ] Ratio compresión promedio: {ratio_prom:.3f}")
        assert ratio_prom < 1.0, "El texto no debería crecer tras limpiar"

    def test_no_pierde_palabras_significativas(self, texto_con_bom, texto_espacios_dobles):
        """
        Toda palabra de >= 3 chars de la entrada
        debe aparecer en la salida (case-insensitive).
        """
        for t in [texto_con_bom, texto_espacios_dobles]:
            limpiado = limpiar(t).lower()
            # El BOM (\ufeff) está entre los caracteres que clean DEBE eliminar,
            # así que no se busca en la salida; el resto de tokens sí.
            palabras = [w.lstrip("\ufeff") for w in t.lower().split() if len(w.lstrip("\ufeff")) >= 3]
            perdidas = [w for w in palabras if w not in limpiado]
            print(f"\n[ MÉTRICA ] Palabras perdidas: {len(perdidas)}/{len(palabras)}")
            assert len(perdidas) == 0, f"Palabras perdidas: {perdidas}"


 
# Test con datos del dominio (opcional)

class TestCleanDominio:
    """
    Con el corpus descargado, se puede cargar un PDF del dominio
    y verificar que clean no rompe tablas de tarifas ni listas.
    PDF requerido para este test: data/PreciosPublicos2026.pdf
    """

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "data" / "PreciosPublicos2026.pdf").exists(),
        reason="Necesita el corpus descargado en data/"
    )
    def test_pdf_real_mantiene_tablas(self):
        from src.load import cargar_archivos
        from src.clean import limpiar

        docs = cargar_archivos("./data/PreciosPublicos2026.pdf")
        assert len(docs) > 0

        # Verifica que al limpiar no se rompe una fila de tabla
        texto_crudo = docs[0].page_content
        texto_limpio = limpiar(texto_crudo)

        # Verifica que las tarifas que tiene el PDF se conservan
        numeros_crudos = re.findall(r'\d+[.,]?\d*', texto_crudo)
        numeros_limpios = re.findall(r'\d+[.,]?\d*', texto_limpio)
        assert len(numeros_limpios) >= len(numeros_crudos) * 0.9, \
            "Se perdieron demasiados números (¿tarifas rotas?)"