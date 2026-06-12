from langchain_core.messages import HumanMessage

from prometheus.lang_graph.nodes.reset_messages_node import ResetMessagesNode


def test_reset_messages_node():
    reset_build_messages_node = ResetMessagesNode("build_messages")

    state = {
        "build_messages": [HumanMessage(content="message 1"), HumanMessage(content="message 2")],
        "test_messages": [HumanMessage(content="message 3"), HumanMessage(content="message 4")],
    }

    reset_build_messages_node(state)

    assert "build_messages" in state
    assert len(state["build_messages"]) == 0

    assert "test_messages" in state
    assert len(state["test_messages"]) == 2
