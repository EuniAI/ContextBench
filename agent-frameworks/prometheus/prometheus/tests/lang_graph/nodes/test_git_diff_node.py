from unittest.mock import Mock

import pytest

from prometheus.git.git_repository import GitRepository
from prometheus.lang_graph.nodes.git_diff_node import GitDiffNode


@pytest.fixture
def mock_git_repo():
    git_repo = Mock(spec=GitRepository)
    git_repo.get_diff.return_value = "sample diff content"
    return git_repo


def test_git_diff_node(mock_git_repo):
    node = GitDiffNode(mock_git_repo, "patch")

    # Execute
    result = node({})

    # Assert
    assert result == {"patch": "sample diff content"}
    mock_git_repo.get_diff.assert_called_with(None)


def test_git_diff_node_with_excluded_files(mock_git_repo):
    node = GitDiffNode(mock_git_repo, "patch", "excluded_file")

    # Execute
    result = node({"excluded_file": "/foo/bar.py"})

    # Assert
    assert result == {"patch": "sample diff content"}
    mock_git_repo.get_diff.assert_called_with(["/foo/bar.py"])
