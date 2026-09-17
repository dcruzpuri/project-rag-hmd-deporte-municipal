"""
tests/test_load.py
Tests offline unitarios para load.py (hash de archivo, detección de encoding y loaders CSV/TXT).
Ejecutar:  pytest tests/test_load.py -v
"""
import hashlib
import sys
from pathlib import Path
from typing import Callable

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.load import _detectar_encoding, _hash_file, _load_csv


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Callable[[bytes], str]:
    """Devuelve ayudante para crear CSVs con los bytes dados."""

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


class TestLoadSource:
    """source = basename limpio en TODOS los loaders y en cargar_archivos:
    sin prefijo de carpeta ni ruta con separador del SO (dedup/id/coherencia
    de metadata por clave de fuente)."""

    def test_load_txt_ruta_absoluta_source_es_basename(self, tmp_csv: Callable[[bytes], str]) -> None:
        from src.load import _load_text

        ruta = tmp_csv("Piscina: 5 €/día.\n".encode("utf-8"), name="fixture.txt")
        docs = _load_text(ruta)
        assert all(d.metadata["source"] == "fixture.txt" for d in docs)

    def test_load_csv_ruta_absoluta_source_es_basename(self, tmp_csv: Callable[[bytes], str]) -> None:
        ruta = tmp_csv("nombre\nCalle Añaña\n".encode("utf-8"), name="fixture2.csv")
        docs = _load_csv(ruta)
        assert all(d.metadata["source"] == "fixture2.csv" for d in docs)

    def test_cargar_archivos_ruta_absoluta_source_es_basename(self, tmp_csv: Callable[[bytes], str]) -> None:
        from src.load import cargar_archivos

        ruta = tmp_csv("contenido prueba\n".encode("utf-8"), name="fixture3.txt")
        docs = cargar_archivos([ruta])
        assert {d.metadata["source"] for d in docs} == {"fixture3.txt"}

    def test_cargar_archivos_dir_sin_sep_en_source(self, tmp_csv: Callable[[bytes], str]) -> None:
        from src.load import cargar_archivos

        ruta = tmp_csv("contenido prueba\n".encode("utf-8"), name="fixture4.txt")
        docs = cargar_archivos([Path(ruta).parent])
        assert all(d.metadata["source"] == "fixture4.txt" for d in docs)


class TestFileHash:
    """metadata.file_hash = SHA-256 del archivo crudo, único y estable."""

    def test_hash_coincide_con_sha256_cruzo(self, tmp_csv: Callable[[bytes], str]) -> None:
        datos = "Piscina: 5 €/día; abono: 40 €.\n".encode("utf-8")
        ruta = tmp_csv(datos, name="fixture5.txt")
        assert _hash_file(ruta) == hashlib.sha256(datos).hexdigest()

    def test_cargar_txt_file_hash_siempre(self, tmp_csv: Callable[[bytes], str]) -> None:
        from src.load import _load_text

        ruta = tmp_csv("contenido de prueba\n".encode("utf-8"), name="fixture6.txt")
        docs = _load_text(ruta)
        esperado = hashlib.sha256(Path(ruta).read_bytes()).hexdigest()
        assert all(d.metadata["file_hash"] == esperado for d in docs)

    def test_cargar_csv_file_hash_propagado_a_filas(self, tmp_csv: Callable[[bytes], str]) -> None:
        # Un CSV genera muchos documentos a partir de un único archivo:
        # la identificación debe compartir el hash del origen en todos.
        ruta = tmp_csv("nombre\nCalle Añaña\n".encode("utf-8"), name="fixture7.csv")
        docs = _load_csv(ruta)
        esperado = hashlib.sha256(Path(ruta).read_bytes()).hexdigest()
        assert docs
        assert {d.metadata["file_hash"] for d in docs} == {esperado}

    def test_hash_unico_por_contenido(self, tmp_csv: Callable[[bytes], str]) -> None:
        a = tmp_csv("contenido A\n".encode("utf-8"), name="fixture8.txt")
        b = tmp_csv("contenido B\n".encode("utf-8"), name="fixture9.txt")
        assert _hash_file(a) != _hash_file(b)

    def test_hash_estable_entre_lecturas(self, tmp_csv: Callable[[bytes], str]) -> None:
        ruta = tmp_csv("contenido estable\n".encode("utf-8"), name="fixture10.txt")
        assert _hash_file(ruta) == _hash_file(ruta)
