from unittest.mock import Mock

import pytest

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.subgraphs.build_and_test_subgraph import BuildAndTestSubgraph
from tests.test_utils.util import FakeListChatWithToolsModel


@pytest.fixture
def mock_container():
    return Mock(spec=BaseContainer)


@pytest.fixture
def mock_kg():
    return Mock(spec=KnowledgeGraph)


def test_build_and_test_subgraph_basic_initialization(mock_container, mock_kg):
    """Test that BuildAndTestSubgraph initializes correctly with basic components."""
    # Initialize fake model with empty responses
    fake_model = FakeListChatWithToolsModel(responses=[])

    # Initialize the subgraph
    subgraph = BuildAndTestSubgraph(container=mock_container, model=fake_model, kg=mock_kg)

    # Verify the subgraph was created
    assert subgraph.subgraph is not None


def test_build_and_test_subgraph_with_commands(mock_container, mock_kg):
    """Test that BuildAndTestSubgraph initializes correctly with build and test commands."""
    fake_model = FakeListChatWithToolsModel(responses=[])
    build_commands = ["make build"]
    test_commands = ["make test"]

    subgraph = BuildAndTestSubgraph(
        container=mock_container,
        model=fake_model,
        kg=mock_kg,
        build_commands=build_commands,
        test_commands=test_commands,
    )

    assert subgraph.subgraph is not None
