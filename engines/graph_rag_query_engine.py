from llama_index.core.query_engine import CustomQueryEngine
import re


class GraphRAGQueryEngine(CustomQueryEngine):
    def __init__(self, graph_store, llm):
        self.graph_store = graph_store
        self.llm = llm

    def query(self, query_str):
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [self.generate_answer_from_summary(summary, query_str) for summary in community_summaries.values()]
        return self.aggregate_answers(community_answers)

    def generate_answer_from_summary(self, community_summary, query):
        prompt = f"Given the summary: {community_summary}, answer the query: {query}"
        response = self.llm.chat([{"role": "system", "content": prompt}])
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

    def aggregate_answers(self, community_answers):
        prompt = f"Aggregate the following answers: {community_answers}"
        response = self.llm.chat([{"role": "system", "content": prompt}])
        return re.sub(r"^assistant:\s*", "", str(response)).strip()
