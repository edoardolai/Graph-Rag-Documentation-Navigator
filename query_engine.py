from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import os
from templates import QA_TEMPLATE, PREFIX, CYPHER_EXAMPLES


def create_code_explorer_rag():
    """Create a Graph RAG system for exploring the CodeCarbon codebase."""
    # Connect to Neo4j
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    # Create LLM
    llm = ChatOpenAI(
        temperature=0, model="gpt-4o-mini", api_key=os.getenv("OPEN_AI_API_KEY")
    )

    cypher_examples = CYPHER_EXAMPLES
    # Create the example prompt template
    example_prompt = PromptTemplate.from_template(
        "User question: {question}\nCypher query: {query}"
    )

    # Create the few-shot prompt for Cypher generation with precise file targeting
    cypher_prompt = FewShotPromptTemplate(
        examples=cypher_examples,
        example_prompt=example_prompt,
        prefix=PREFIX,
        suffix="User question: {question}\nCypher query: ",
        input_variables=["schema", "question"],
    )

    # Create custom QA template focused on CodeCarbon functionality with file context
    qa_template = QA_TEMPLATE

    qa_prompt = PromptTemplate.from_template(qa_template)

    # Create the QA chain
    chain = GraphCypherQAChain.from_llm(
        graph=graph,
        llm=llm,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        verbose=True,
        validate_cypher=True,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        max_page_size=999999,
    )

    return chain


def query_codebase(question):
    """Query the CodeCarbon codebase with a natural language question."""
    chain = create_code_explorer_rag()

    try:
        result = chain.invoke({"query": question})

        # Extract the main answer
        answer = result.get("result", result.get("answer", "No answer found"))

        # Extract the Cypher query that was generated
        query = "No query available"
        context = None
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if isinstance(step, dict) and "query" in step:
                    query = step["query"]
                if isinstance(step, dict) and "context" in step:
                    context = step["context"]

        # Process context into a dictionary of files and their text chunks
        context_by_file = {}
        if context:
            context_by_file = process_context_by_file(context)

        return {
            "answer": answer,
            "query": query,
            "context": context_by_file,
        }

    except Exception as e:
        print(f"Error querying codebase: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "query": "Query failed to execute",
            "context": {},
        }


def process_context_by_file(context):
    """
    Process the context into a dictionary with filenames as keys and
    arrays of text chunks as values.
    """
    file_chunks = {}
    try:
        if isinstance(context, list):
            for item in context:
                if (
                    isinstance(item, dict)
                    and "d.file_path" in item
                    and "tc.text" in item
                ):
                    file_path = item["d.file_path"]
                    text_chunk = item["tc.text"]

                    if file_path:
                        # Extract just the filename from path
                        filename = file_path.split("/")[-1].strip()

                        # Add to the dictionary
                        if filename not in file_chunks:
                            file_chunks[filename] = []
                        file_chunks[filename].append(text_chunk)
        # Handle the legacy string format if needed
        elif isinstance(context, str):
            # This is kept for backward compatibility but will likely not be used
            lines = context.split("\n")
            current_file = "unknown"
            for line in lines:
                if ".py" in line or ".ts" in line or ".tsx" in line or ".html" in line:
                    parts = line.split("|")
                    if len(parts) > 1:
                        file_path = parts[0].strip()
                        content = parts[1].strip() if len(parts) > 1 else ""

                        if "/" in file_path:
                            filename = file_path.split("/")[-1].strip()
                        else:
                            filename = file_path

                        if filename not in file_chunks:
                            file_chunks[filename] = []
                        file_chunks[filename].append(content)
                        current_file = filename
                else:
                    # Append to the most recent file if content continues
                    if current_file in file_chunks and file_chunks[current_file]:
                        file_chunks[current_file][-1] += "\n" + line

    except Exception as e:
        print(f"Error processing context by file: {e}")

    return file_chunks


# You can keep this function if other parts of your code depend on it,
# otherwise it can be removed as it's replaced by process_context_by_file
def extract_files_from_context(context):
    """Extract all unique file names mentioned in the context"""
    if isinstance(context, list):
        context_dict = process_context_by_file(context)
        return list(context_dict.keys())

    files = set()
    try:
        # Handle context as a string (for backward compatibility)
        if isinstance(context, str):
            lines = context.split("\n")
            for line in lines:
                if ".py" in line or ".ts" in line or ".tsx" in line or ".html" in line:
                    parts = line.split("|")
                    if len(parts) > 1:
                        file_path = parts[0].strip()
                        if "/" in file_path:
                            filename = file_path.split("/")[-1].strip()
                            files.add(filename)
                        else:
                            files.add(file_path)
    except Exception as e:
        print(f"Error extracting files from context: {e}")

    return list(files)
