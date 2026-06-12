from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.general_build_node import GeneralBuildNode
from prometheus.lang_graph.subgraphs.build_and_test_state import BuildAndTestState
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_container():
    return Mock(spec=BaseContainer)


@pytest.fixture
def mock_kg():
    kg = Mock(spec=KnowledgeGraph)
    kg.get_file_tree.return_value = ".\n├── src\n│   └── main.py\n└── build.gradle"
    return kg


@pytest.fixture
def fake_llm():
    return FakeListChatWithToolsModel(responses=["Build command executed successfully"])


def test_format_human_message_basic(mock_container, mock_kg, fake_llm):
    """Test basic human message formatting."""
    node = GeneralBuildNode(fake_llm, mock_container, mock_kg)
    state = BuildAndTestState({})

    message = node.format_human_message(state)

    assert isinstance(message, HumanMessage)
    assert "project structure is:" in message.content
    assert mock_kg.get_file_tree() in message.content


def test_format_human_message_with_build_summary(mock_container, mock_kg, fake_llm):
    """Test message formatting with build command summary."""
    node = GeneralBuildNode(fake_llm, mock_container, mock_kg)
    state = BuildAndTestState({"build_command_summary": "Previous build used gradle"})

    message = node.format_human_message(state)

    assert "Previous build used gradle" in message.content
    assert "The previous build summary is:" in message.content


def test_call_method_with_no_build(mock_container, mock_kg, fake_llm):
    """Test __call__ method when exist_build is False."""
    node = GeneralBuildNode(fake_llm, mock_container, mock_kg)
    state = BuildAndTestState({"exist_build": False})

    result = node(state)

    assert "build_messages" in result
    assert len(result["build_messages"]) == 1
    assert (
        "Previous agent determined there is no build system" in result["build_messages"][0].content
    )
