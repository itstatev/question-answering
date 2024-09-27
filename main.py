from loaders.pdf_loader import PDFLoader
from extractors.graph_rag_extractor import GraphRAGExtractor
from stores.graph_rag_store import GraphRAGStore
from engines.graph_rag_query_engine import GraphRAGQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from utils.helpers import parse_fn  

import os


def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(model="gpt-3.5-turbo")

    pdf_paths = ["/home/tatev/Desktop/AUA Statistics/PSS/PSS_1.pdf"]  
    pdf_loader = PDFLoader(pdf_paths)
    documents = pdf_loader.load_pdfs()

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)

    extract_prompt = "Extract entities and relationships from the following text: {text}"
    extractor = GraphRAGExtractor(llm=llm, extract_prompt=extract_prompt, parse_fn=parse_fn)
    nodes_with_metadata = extractor.extract_triples(nodes)

    graph_store = GraphRAGStore(llm=llm)
    index = VectorStoreIndex(nodes=nodes_with_metadata, property_graph_store=graph_store)
    graph_store.build_communities()

    query_engine = GraphRAGQueryEngine(graph_store=graph_store, llm=llm)
    response = query_engine.query("What are the main topics discussed?")
    print(response)


if __name__ == "__main__":
    main()