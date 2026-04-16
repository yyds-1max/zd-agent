from app.schemas.query import QueryRequest, QueryResponse
from app.services.query_graph_service import QueryGraphService


class QueryPipeline:
    def __init__(self) -> None:
        self.graph = QueryGraphService()

    def run(self, payload: QueryRequest) -> QueryResponse:
        return self.graph.run(payload)
