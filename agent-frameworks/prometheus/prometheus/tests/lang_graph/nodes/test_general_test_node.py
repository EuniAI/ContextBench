from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.general_test_node import GeneralTestNode
from prometheus.lang_graph.subgraphs.build_and_test_state import BuildAndTestState
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_container():
    return Mock(spec=BaseContainer)


@pytest.fixture
def mock_kg():
    kg = Mock(spec=KnowledgeGraph)
    kg.get_file_tree.return_value = "./\n├── tests/\n│   └── test_main.py"
    return kg


@pytest.fixture
def fake_llm():
    return FakeListChatWithToolsModel(responses=["Tests executed successfully"])


@pytest.fixture
def basic_state():
    return BuildAndTestState(
        {"exist_test": True, "test_messages": [], "test_command_summary": None}
    )


def test_format_human_message(mock_container, mock_kg, fake_llm, basic_state):
    """Test basic message formatting."""
    node = GeneralTestNode(fake_llm, mock_container, mock_kg)
    message = node.format_human_message(basic_state)

    assert isinstance(message, HumanMessage)
    assert mock_kg.get_file_tree() in message.content


def test_call_with_no_tests(mock_container, mock_kg, fake_llm):
    """Test behavior when no tests exist."""
    node = GeneralTestNode(fake_llm, mock_container, mock_kg)
    state = BuildAndTestState({"exist_test": False})

    result = node(state)

    assert "build_messages" in result
    assert "no test framework" in result["build_messages"][0].content


def test_call_normal_execution(mock_container, mock_kg, fake_llm, basic_state):
    """Test normal execution flow."""
    node = GeneralTestNode(fake_llm, mock_container, mock_kg)

    result = node(basic_state)

    assert "test_messages" in result
    assert len(result["test_messages"]) == 1
    assert result["test_messages"][0].content == "Tests executed successfully"


def test_format_human_message_with_summary(mock_container, mock_kg, fake_llm):
    """Test message formatting with test summary."""
    node = GeneralTestNode(fake_llm, mock_container, mock_kg)
    state = BuildAndTestState({"test_command_summary": "Previous test used pytest"})

    message = node.format_human_message(state)

    assert "Previous test used pytest" in message.content
