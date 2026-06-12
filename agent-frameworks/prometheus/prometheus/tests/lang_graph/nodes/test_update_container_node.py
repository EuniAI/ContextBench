from pathlib import Path
from unittest.mock import Mock

from prometheus.docker.general_container import GeneralContainer
from prometheus.git.git_repository import GitRepository
from prometheus.lang_graph.nodes.update_container_node import UpdateContainerNode


def test_update_container_node():
    mocked_container = Mock(spec=GeneralContainer)
    mocked_container.is_running.return_value = True
    mocked_git_repo = Mock(spec=GitRepository)
    mocked_git_repo.get_diff.return_value = "--- /dev/null\n+++ b/newfile\n@@ -0,0 +1 @@\n+content"
    mocked_git_repo.get_working_directory.return_value = Path("/test/working/dir/repositories/repo")
    update_container_node = UpdateContainerNode(mocked_container, mocked_git_repo)

    update_container_node(None)

    assert mocked_git_repo.get_diff.call_count == 1
    assert mocked_container.is_running.call_count == 1
    assert mocked_container.update_files.call_count == 1
    mocked_container.update_files.assert_called_with(
        Path("/test/working/dir/repositories/repo"), [Path("newfile")], []
    )
