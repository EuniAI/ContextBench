import functools
from typing import Mapping, Optional, Sequence, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prometheus.docker.base_container import BaseContainer
from prometheus.graph.knowledge_graph import KnowledgeGraph
from prometheus.lang_graph.nodes.general_build_node import GeneralBuildNode
from prometheus.lang_graph.nodes.general_build_structured_node import (
    GeneralBuildStructuredNode,
)
from prometheus.lang_graph.nodes.general_test_node import GeneralTestNode
from prometheus.lang_graph.nodes.general_test_structured_node import GeneralTestStructuredNode
from prometheus.lang_graph.nodes.noop_node import NoopNode
from prometheus.lang_graph.nodes.user_defined_build_node import UserDefinedBuildNode
from prometheus.lang_graph.nodes.user_defined_test_node import UserDefinedTestNode
from prometheus.lang_graph.subgraphs.build_and_test_state import BuildAndTestState


class BuildAndTestSubgraph:
    def __init__(
        self,
        container: BaseContainer,
        model: BaseChatModel,
        kg: KnowledgeGraph,
        build_commands: Optional[Sequence[str]] = None,
        test_commands: Optional[Sequence[str]] = None,
    ):
        build_branch_node = NoopNode()
        if build_commands:
            build_node = UserDefinedBuildNode(container)
        else:
            build_node = GeneralBuildNode(model, container, kg)
            build_tools = ToolNode(
                tools=build_node.tools,
                name="build_tools",
                messages_key="build_messages",
            )
        general_build_structured_node = GeneralBuildStructuredNode(model)

        test_branch_node = NoopNode()
        if test_commands:
            test_node = UserDefinedTestNode(container)
        else:
            test_node = GeneralTestNode(model, container, kg)
            test_tools = ToolNode(
                tools=test_node.tools,
                name="test_tools",
                messages_key="test_messages",
            )
        general_test_structured_node = GeneralTestStructuredNode(model)

        workflow = StateGraph(BuildAndTestState)
        workflow.add_node("build_branch_node", build_branch_node)
        workflow.add_node("build_node", build_node)
        if not build_commands:
            workflow.add_node("build_tools", build_tools)
        workflow.add_node("general_build_structured_node", general_build_structured_node)

        workflow.add_node("test_branch_node", test_branch_node)
        workflow.add_node("test_node", test_node)
        if not test_commands:
            workflow.add_node("test_tools", test_tools)
        workflow.add_node("general_test_structured_node", general_test_structured_node)

        workflow.set_entry_point("build_branch_node")
        workflow.add_conditional_edges(
            "build_branch_node",
            lambda state: state["run_build"],
            {True: "build_node", False: "test_branch_node"},
        )
        if build_commands:
            workflow.add_edge("build_node", "general_build_structured_node")
        else:
            workflow.add_conditional_edges(
                "build_node",
                functools.partial(tools_condition, messages_key="build_messages"),
                {
                    "tools": "build_tools",
                    END: "general_build_structured_node",
                },
            )
            workflow.add_edge("build_tools", "build_node")
        workflow.add_edge("general_build_structured_node", "test_node")

        workflow.add_conditional_edges(
            "test_branch_node",
            lambda state: state["run_existing_test"],
            {True: "test_node", False: END},
        )
        if test_commands:
            workflow.add_edge("test_node", "general_test_structured_node")
        else:
            workflow.add_conditional_edges(
                "test_node",
                functools.partial(tools_condition, messages_key="test_messages"),
                {
                    "tools": "test_tools",
                    END: "general_test_structured_node",
                },
            )
            workflow.add_edge("test_tools", "test_node")
        workflow.add_edge("general_test_structured_node", END)

        self.subgraph = workflow.compile()

    def invoke(
        self,
        run_build: bool,
        run_existing_test: bool,
        exist_build: Optional[bool] = None,
        build_command_summary: Optional[str] = None,
        build_fail_log: Optional[str] = None,
        exist_test: Optional[bool] = None,
        test_command_summary: Optional[str] = None,
        existing_test_fail_log: Optional[str] = None,
    ) -> Mapping[str, Union[bool, str]]:
        config = None

        input_state = {
            "run_build": run_build,
            "run_existing_test": run_existing_test,
        }
        if exist_build:
            input_state["exist_build"] = exist_build
        if build_command_summary:
            input_state["build_command_summary"] = build_command_summary
        if build_fail_log:
            input_state["build_fail_log"] = build_fail_log
        if exist_test:
            input_state["exist_test"] = exist_test
        if test_command_summary:
            input_state["test_command_summary"] = test_command_summary
        if existing_test_fail_log:
            input_state["existing_test_fail_log"] = existing_test_fail_log

        output_state = self.subgraph.invoke(input_state, config)

        return {
            "exist_build": output_state.get("exist_build", False),
            "build_command_summary": output_state.get("build_command_summary", ""),
            "build_fail_log": output_state.get("build_fail_log", ""),
            "exist_test": output_state.get("exist_test", False),
            "test_command_summary": output_state.get("test_command_summary", ""),
            "existing_test_fail_log": output_state.get("existing_test_fail_log", ""),
        }
