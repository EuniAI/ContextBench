from pathlib import Path

from prometheus.utils.patch_util import get_updated_files


def test_get_updated_files_empty_diff():
    diff = ""
    added, modified, removed = get_updated_files(diff)
    assert len(added) == 0
    assert len(modified) == 0
    assert len(removed) == 0


def test_get_updated_files_added_only():
    diff = """
diff --git a/new_file.txt b/new_file.txt
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/new_file.txt
@@ -0,0 +1 @@
+New content
"""
    added, modified, removed = get_updated_files(diff)
    assert len(added) == 1
    assert len(modified) == 0
    assert len(removed) == 0
    assert added[0] == Path("new_file.txt")


def test_get_updated_files_modified_only():
    diff = """
diff --git a/modified_file.txt b/modified_file.txt
index 1234567..89abcdef
--- a/modified_file.txt
+++ b/modified_file.txt
@@ -1 +1 @@
-Old content
+Modified content
"""
    added, modified, removed = get_updated_files(diff)
    assert len(added) == 0
    assert len(modified) == 1
    assert len(removed) == 0
    assert modified[0] == Path("modified_file.txt")


def test_get_updated_files_removed_only():
    diff = """
diff --git a/removed_file.txt b/removed_file.txt
deleted file mode 100644
index 1234567..0000000
--- a/removed_file.txt
+++ /dev/null
@@ -1 +0,0 @@
-Content to be removed
"""
    added, modified, removed = get_updated_files(diff)
    assert len(added) == 0
    assert len(modified) == 0
    assert len(removed) == 1
    assert removed[0] == Path("removed_file.txt")


def test_get_updated_files_multiple_changes():
    diff = """
diff --git a/new_file.txt b/new_file.txt
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/new_file.txt
@@ -0,0 +1 @@
+New content
diff --git a/modified_file.txt b/modified_file.txt
index 1234567..89abcdef
--- a/modified_file.txt
+++ b/modified_file.txt
@@ -1 +1 @@
-Old content
+Modified content
diff --git a/removed_file.txt b/removed_file.txt
deleted file mode 100644
index 1234567..0000000
--- a/removed_file.txt
+++ /dev/null
@@ -1 +0,0 @@
-Content to be removed
"""
    added, modified, removed = get_updated_files(diff)
    assert len(added) == 1
    assert len(modified) == 1
    assert len(removed) == 1
    assert added[0] == Path("new_file.txt")
    assert modified[0] == Path("modified_file.txt")
    assert removed[0] == Path("removed_file.txt")


def test_get_updated_files_with_subfolders():
    diff = """
diff --git a/folder1/new_file.txt b/folder1/new_file.txt
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/folder1/new_file.txt
@@ -0,0 +1 @@
+New content
diff --git a/folder2/subfolder/modified_file.txt b/folder2/subfolder/modified_file.txt
index 1234567..89abcdef
--- a/folder2/subfolder/modified_file.txt
+++ b/folder2/subfolder/modified_file.txt
@@ -1 +1 @@
-Old content
+Modified content
"""
    added, modified, removed = get_updated_files(diff)
    assert len(added) == 1
    assert len(modified) == 1
    assert len(removed) == 0
    assert added[0] == Path("folder1/new_file.txt")
    assert modified[0] == Path("folder2/subfolder/modified_file.txt")
