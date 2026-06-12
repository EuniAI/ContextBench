from unittest.mock import Mock

import pytest
from langchain_core.messages import ToolMessage

from prometheus.docker.base_container import BaseContainer
from prometheus.lang_graph.nodes.user_defined_build_node import UserDefinedBuildNode


@pytest.fixture
def mock_container():
    container = Mock(spec=BaseContainer)
    container.run_build.return_value = "Build successful"
    return container


@pytest.fixture
def build_node(mock_container):
    return UserDefinedBuildNode(container=mock_container)


def test_successful_build(build_node, mock_container):
    expected_output = "Build successful"

    result = build_node(None)

    assert isinstance(result, dict)
    assert "build_messages" in result
    assert len(result["build_messages"]) == 1
    assert isinstance(result["build_messages"][0], ToolMessage)
    assert result["build_messages"][0].content == expected_output
    mock_container.run_build.assert_called_once()
