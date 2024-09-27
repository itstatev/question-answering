from llama_index.core.graph_stores.types import EntityNode, Relation  
from llama_index.core.llms import ChatMessage

class GraphRAGExtractor:
    def __init__(self, llm, extract_prompt, parse_fn, max_paths_per_chunk=2):
        self.llm = llm
        self.extract_prompt = extract_prompt
        self.parse_fn = parse_fn
        self.max_paths_per_chunk = max_paths_per_chunk

    def extract_triples(self, nodes):
        for node in nodes:
            text = node.text
            prompt = self.extract_prompt.format(text=text)

            messages = [
                ChatMessage(role="system", content="You are an entity and relationship extractor."),
                ChatMessage(role="user", content=prompt)
            ]

            response = self.llm.chat(messages)

            entities, relationships = self.parse_fn(response)
            self.add_metadata_to_node(node, entities, relationships)
        return nodes

    def add_metadata_to_node(self, node, entities, relationships):
        existing_nodes = []
        existing_relations = []
        metadata = node.metadata.copy()

        for entity, entity_type, description in entities:
            entity_node = EntityNode(name=entity, label=entity_type, properties=metadata)
            existing_nodes.append(entity_node)

        for subj, rel, obj, description in relationships:
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            rel_node = Relation(label=rel, source_id=subj_node.id, target_id=obj_node.id, properties=metadata)
            existing_relations.append(rel_node)

        node.metadata["KG_NODES"] = existing_nodes
        node.metadata["KG_RELATIONS"] = existing_relations
