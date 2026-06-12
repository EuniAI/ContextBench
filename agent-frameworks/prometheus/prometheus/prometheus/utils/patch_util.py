from pathlib import Path
from typing import Sequence, Tuple

from unidiff import PatchSet


def get_updated_files(diff: str) -> Tuple[Sequence[Path], Sequence[Path], Sequence[Path]]:
    patch = PatchSet(diff)
    added_files = []
    modified_files = []
    removed_files = []

    for added_file in patch.added_files:
        added_files.append(Path(added_file.path))

    for modified_file in patch.modified_files:
        modified_files.append(Path(modified_file.path))

    for removed_file in patch.removed_files:
        removed_files.append(Path(removed_file.path))

    return added_files, modified_files, removed_files
