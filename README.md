# Graph-RAG Documentation Navigator

## Overview

This project is a Proof of Concept (POC) for a Graph-based Retrieval-Augmented Generation (Graph-RAG) system designed to help developers navigate poorly documented codebases. Unlike traditional RAG systems that work with unstructured text, Graph-RAG structures code relationships into a knowledge graph, allowing for more contextual and accurate retrieval when answering questions about the code.

We built this project while learning about Graph-RAG for the first time, making it a valuable learning experience that helped us understand how knowledge graphs can enhance code exploration and documentation.

### What is Graph-RAG?

Graph-RAG enhances traditional RAG by:

1. **Structuring knowledge** into a graph with linked entities
2. **Retrieving information** based on relationships rather than simple keyword matching
3. **Capturing dependencies** between code components, improving accuracy and context

## Example Use Case

Given a complex, poorly documented codebase, a developer might ask:

> "Where is the authentication functionality implemented?"

The system will:

1. Convert this natural language question into a Cypher query
2. Search the Neo4j graph for relevant code components
3. Return specific file snippets with explanations of how they relate to authentication

## Tech Stack

- **Python**: Core programming language
- **Neo4j**: Graph database for storing code structure and relationships
- **LangChain**: Orchestration framework connecting LLMs with the graph database
- **OpenAI API**: Powers natural language understanding and Cypher query generation
- **Streamlit**: Provides the web interface for interacting with the system

## Project Structure

The system consists of four main components:

1. **Input Preparation** (`repo_cleaner.py`): Cleans and organizes source files
2. **Graph Construction** (`main.py`): Builds relationships in Neo4j
3. **Query Engine** (`query_engine.py`): Handles LLM-powered Cypher queries
4. **Web Interface** (`app.py`): Streamlit-based UI for interaction

## Setup Instructions

### Prerequisites

- Python 3.8+
- Neo4j Desktop installed
- OpenAI API key
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/graph-rag-documentation-navigator.git
cd graph-rag-documentation-navigator
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set up Neo4j

1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new database and start it
3. Note your connection details (URL, username, and password)

### Step 4: Clone a repository to analyze

```bash
git clone https://github.com/example/repo-to-analyze.git
```

### Step 5: Configure your environment

1. Copy the `.env.template` file to create a `.env` file:

```bash
cp .env.template .env
```

2. Edit the `.env` file with your details:

```
NEO4J_URL = bolt://localhost:7687
NEO4J_USERNAME = neo4j
NEO4J_PASSWORD = your_password
OPEN_AI_API_KEY = your_openai_api_key
RAW_REPO_DIRECTORY_PATH = /path/to/cloned/repository
CLEANED_REPO_DIRECTORY_PATH = /path/to/store/cleaned/repository
```

### Step 6: Clean the repository

```bash
python repo_cleaner.py
```

This script will filter the repository to include only relevant code files:
repo_cleaner.py

```
#relevant code files defaulted to this list, adjust as needed in your case
    included_file_types = {".py", ".js", ".html", ".css", ".ts", ".tsx"}
```

### Step 7: Customize templates.py

The `templates.py` file contains example queries that should be updated for your specific repository:

1. Replace the example queries in `CYPHER_EXAMPLES` with queries relevant to your codebase
2. Ensure `PREFIX` and `QA_TEMPLATE` match your project's domain

### Step 8: Build the knowledge graph

```bash
python main.py
```

This process might take some time as it:

- Cleans the repository
- Creates document nodes
- Chunks text content
- Extracts entities using LLMs
- Builds relationships

### Step 9: Start the web interface

```bash
python -m streamlit run cli_interface.py

#run streamlit as python module, else it might file due to import errors
```

## Usage

Once the web interface is running:

1. Navigate to the provided local URL (typically `http://localhost:8501`)
2. Enter natural language questions about the codebase in the search box
3. View the generated Cypher query, answer, and relevant code context

## Limitations and Future Improvements

As this is a POC created during my learning journey with Graph-RAG, there are several areas for improvement:

- **Enhanced Agentic Behavior**: Implement more advanced agentic capabilities using the LangChain ecosystem
- **Hybrid Retrieval**: Combine graph-based and vector-based retrieval for better results
- **Multi-Repository Support**: Expand to handle multiple codebases simultaneously
- **Chat History**: Maintain conversation context for follow-up questions
- **Performance Optimization**: Improve query efficiency for larger codebases

## Acknowledgments

This project was inspired by various works on Retrieval-Augmented Generation and knowledge graphs. We built it as a learning exercise to better understand these technologies and their applications in code documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
