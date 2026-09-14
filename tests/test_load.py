"""
tests/test_load.py
Tests offline unitarios para load.py (detección de encoding y loaders CSV/TXT).
Ejecutar:  pytest tests/test_load.py -v
"""
import sys
from pathlib import Path
from typing import Callable

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.load import _detectar_encoding, _load_csv


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Callable[[bytes], str]:
    """Devuelve un helper para crear CSVs con los bytes dados."""

    def _h(data: bytes, name: str = "fixture.csv") -> str:
        p = tmp_path / name
        p.write_bytes(data)
        return str(p)

    return _h


class TestDetectarEncoding:
    def test_utf8_puro(self, tmp_csv: Callable[[bytes], str]) -> None:
        ruta = tmp_csv("nómina;precio\nCalle Añaña;10\n".encode("utf-8"))
        assert _detectar_encoding(ruta) == "utf-8-sig"

    def test_utf8_con_bom(self, tmp_csv: Callable[[bytes], str]) -> None:
        # BOM + UTF-8: debe seguir detectarse como utf-8-sig.
        datos = "nombre\nCalle Añaña\n".encode("utf-8-sig")
        ruta = tmp_csv(datos)
        assert _detectar_encoding(ruta) == "utf-8-sig"

    def test_cp1252(self, tmp_csv: Callable[[bytes], str]) -> None:
        # 'á' = 0xe1 no es UTF-8 válido en secuencia: utf-8 falla, cp1252 sí.
        ruta = tmp_csv("nombre\nCalle Añaña 10\n".encode("cp1252"))
        assert _detectar_encoding(ruta) == "cp1252"

    def test_fallback_latin1(self, tmp_csv: Callable[[bytes], str]) -> None:
        # 0x90 no existe ni en cp1252 (undefined): solo latin-1 lo lee.
        ruta = tmp_csv(b"nombre\nF\x90LIX RUBIO\n")
        assert _detectar_encoding(ruta) == "latin-1"

    def test_siempre_devuelve_encoding_valido(self, tmp_csv: Callable[[bytes], str]) -> None:
        p = tmp_csv(bytes(range(256)))
        assert _detectar_encoding(p) in {"utf-8-sig", "cp1252", "latin-1"}


class TestLoadCsv:
    def test_csv_cp1252_carga_sin_error(self, tmp_csv: Callable[[bytes], str]) -> None:
        ruta = tmp_csv("nombre\nCalle Añaña 10\n".encode("cp1252"))
        docs = _load_csv(ruta)
        assert len(docs) == 1
        assert "Calle Añaña 10" in docs[0].page_content

    def test_csv_bom_sin_caracter_extra(self, tmp_csv: Callable[[bytes], str]) -> None:
        ruta = tmp_csv("nombre\nCalle Añaña\n".encode("utf-8-sig"))
        docs = _load_csv(ruta)
        assert not docs[0].page_content.startswith("\ufeff")
        assert "Calle Añaña" in docs[0].page_content

    def test_csv_latin1_corrupto(self, tmp_csv: Callable[[bytes], str]) -> None:
        # Bytes que ni cp1252 ni UTF-8 pueden leer: latin-1 lo rescata.
        ruta = tmp_csv(b"nombre\nF\x90LIX RUBIO\n")
        docs = _load_csv(ruta)
        assert len(docs) >= 1


class TestLoadText:
    def test_txt_cp1252(self, tmp_csv: Callable[[bytes], str]) -> None:
        from src.load import _load_text

        ruta = tmp_csv("Piscina: 5 €/día; abono: 40 €.\n".encode("cp1252"), name="fixture.txt")
        docs = _load_text(ruta)
        assert len(docs) == 1
        assert "Piscina" in docs[0].page_content
