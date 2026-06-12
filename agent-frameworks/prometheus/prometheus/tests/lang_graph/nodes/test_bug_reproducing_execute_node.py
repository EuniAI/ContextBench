from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from prometheus.docker.base_container import BaseContainer
from prometheus.lang_graph.nodes.bug_reproducing_execute_node import BugReproducingExecuteNode
from prometheus.lang_graph.subgraphs.bug_reproduction_state import BugReproductionState
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_container():
    return Mock(spec=BaseContainer)


@pytest.fixture
def test_state():
    return BugReproductionState(
        {
            "issue_title": "Test Bug",
            "issue_body": "Bug description",
            "issue_comments": ["Comment 1", "Comment 2"],
            "bug_context": "Context of the bug",
            "bug_reproducing_write_messages": [AIMessage(content="patch")],
            "bug_reproducing_file_messages": [AIMessage(content="path")],
            "bug_reproducing_execute_messages": [],
            "bug_reproducing_patch": "--- /dev/null\n+++ b/newfile\n@@ -0,0 +1 @@\n+content",
        }
    )


def test_format_human_message_with_test_commands(mock_container, test_state):
    """Test message formatting with provided test commands."""
    fake_llm = FakeListChatWithToolsModel(responses=["test"])
    test_commands = ["pytest", "python -m unittest"]
    node = BugReproducingExecuteNode(fake_llm, mock_container, test_commands)

    message = node.format_human_message(test_state, "/foo/bar/test.py")

    assert isinstance(message, HumanMessage)
    assert "Test Bug" in message.content
    assert "Bug description" in message.content
    assert "Comment 1" in message.content
    assert "Comment 2" in message.content
    assert "pytest" in message.content
    assert "/foo/bar/test.py" in message.content


def test_format_human_message_without_test_commands(mock_container, test_state):
    """Test message formatting without test commands."""
    fake_llm = FakeListChatWithToolsModel(responses=["test"])
    node = BugReproducingExecuteNode(fake_llm, mock_container)

    message = node.format_human_message(test_state, "/foo/bar/test.py")

    assert isinstance(message, HumanMessage)
    assert "Test Bug" in message.content
    assert "Bug description" in message.content
    assert "User provided test commands:\n" in message.content
    assert "/foo/bar/test.py" in message.content


def test_call_method(mock_container, test_state):
    """Test the __call__ method execution."""
    fake_response = "Test execution completed"
    fake_llm = FakeListChatWithToolsModel(responses=[fake_response])
    node = BugReproducingExecuteNode(fake_llm, mock_container)

    result = node(test_state)

    assert "bug_reproducing_execute_messages" in result
    assert len(result["bug_reproducing_execute_messages"]) == 1
    assert result["bug_reproducing_execute_messages"][0].content == fake_response
