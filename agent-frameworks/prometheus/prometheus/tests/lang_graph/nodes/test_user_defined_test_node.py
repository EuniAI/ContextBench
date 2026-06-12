from unittest.mock import Mock

import pytest
from langchain_core.messages import ToolMessage

from prometheus.docker.base_container import BaseContainer
from prometheus.lang_graph.nodes.user_defined_test_node import UserDefinedTestNode


@pytest.fixture
def mock_container():
    container = Mock(spec=BaseContainer)
    container.run_test.return_value = "Test successful"
    return container


@pytest.fixture
def test_node(mock_container):
    return UserDefinedTestNode(container=mock_container)


def test_successful_test(test_node, mock_container):
    expected_output = "Test successful"

    result = test_node(None)

    assert isinstance(result, dict)
    assert "test_messages" in result
    assert len(result["test_messages"]) == 1
    assert isinstance(result["test_messages"][0], ToolMessage)
    assert result["test_messages"][0].content == expected_output
    mock_container.run_test.assert_called_once()
