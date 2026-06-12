from pathlib import Path

TEST_PROJECT_PATH = Path(__file__).parent.parent / "test_project"
GIT_DIR = TEST_PROJECT_PATH / ".dummy_git"
C_FILE = TEST_PROJECT_PATH / "test.c"
BAR_DIR = TEST_PROJECT_PATH / "bar"
JAVA_FILE = BAR_DIR / "test.java"
PYTHON_FILE = BAR_DIR / "test.py"
FOO_DIR = TEST_PROJECT_PATH / "foo"
MD_FILE = FOO_DIR / "test.md"
DUMMY_FILE = FOO_DIR / "test.dummy"
