from langchain_core.language_models.fake_chat_models import FakeListChatModel
from testcontainers.neo4j import Neo4jContainer


class FakeListChatWithToolsModel(FakeListChatModel):
    def bind_tools(self, tools=None, tool_choice=None, **kwargs):
        return self


def clean_neo4j_container(neo4j_container: Neo4jContainer):
    with neo4j_container.get_driver() as driver:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
