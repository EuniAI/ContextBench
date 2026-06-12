from pathlib import Path

import pytest

from prometheus.parser.file_types import FileType


@pytest.mark.parametrize(
    "file_path,expected_type",
    [
        ("script.sh", FileType.BASH),
        ("program.c", FileType.C),
        ("app.cs", FileType.CSHARP),
        ("class.cpp", FileType.CPP),
        ("header.cc", FileType.CPP),
        ("file.cxx", FileType.CPP),
        ("main.go", FileType.GO),
        ("Class.java", FileType.JAVA),
        ("app.js", FileType.JAVASCRIPT),
        ("Service.kt", FileType.KOTLIN),
        ("index.php", FileType.PHP),
        ("script.py", FileType.PYTHON),
        ("query.sql", FileType.SQL),
        ("config.yaml", FileType.YAML),
        ("docker-compose.yml", FileType.YAML),
        ("readme.md", FileType.UNKNOWN),
        ("Makefile", FileType.UNKNOWN),
        ("", FileType.UNKNOWN),
    ],
)
def test_file_type_from_path(file_path: str, expected_type: FileType):
    """Test that file extensions are correctly mapped to FileTypes"""
    path = Path(file_path)
    assert FileType.from_path(path) == expected_type
