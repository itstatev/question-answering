# **GraphRAG PDF Knowledge Graph**

This project builds a **Graph-based Retrieval Augmented Generation (GraphRAG)** pipeline that processes PDF documents, extracts entities and relationships, constructs a knowledge graph, applies community detection, and allows users to query the graph for insights.

## **Key Features:**
- **Load and Parse PDFs:** Extract text from PDF documents.
- **Entity and Relationship Extraction:** Use an AI model to extract entities and their relationships.
- **Knowledge Graph Construction:** Build a knowledge graph from the extracted entities and relationships.
- **Community Detection:** Group related entities into communities using the Hierarchical Leiden algorithm.
- **Query Engine:** Query the graph to retrieve detailed insights about the content.
