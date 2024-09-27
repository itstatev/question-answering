from llama_index.core.graph_stores import SimplePropertyGraphStore
from graspologic.partition import hierarchical_leiden
import re


class GraphRAGStore(SimplePropertyGraphStore):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.community_summaries = {}

    def build_communities(self):
        nx_graph = self._create_nx_graph()
        clusters = hierarchical_leiden(nx_graph, max_cluster_size=5)
        community_info = self._collect_community_info(nx_graph, clusters)
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        import networkx as nx
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))
        for relation in self.graph.relations.values():
            nx_graph.add_edge(relation.source_id, relation.target_id, relationship=relation.label, description=relation.properties["relationship_description"])
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        community_mapping = {item.node: item.cluster for item in clusters}
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in community_info:
                community_info[cluster_id] = []

            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail)
        return community_info

    def _summarize_communities(self, community_info):
        for community_id, details in community_info.items():
            details_text = "\n".join(details)
            summary = self.generate_community_summary(details_text)
            self.community_summaries[community_id] = summary

    def generate_community_summary(self, text):
        prompt = f"Summarize the following relationships: {text}"
        response = self.llm.chat([{"role": "system", "content": prompt}])
        return re.sub(r"^assistant:\s*", "", str(response)).strip()
