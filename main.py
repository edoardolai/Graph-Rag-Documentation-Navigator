import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, PythonLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from tqdm import tqdm
import uuid
import time
from openai import LengthFinishReasonError

# Load environment variables
load_dotenv()

# Path to repository
repo_path = "/Users/edoardo/Graph-Rag-Documentation-Navigator/codecarbon_cleaned"

# Map file extensions to languages for better chunking
EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
}


def clear_database(graph):
    """Clear the existing database to start fresh"""
    print("Clearing existing database...")
    graph.query("MATCH (n) DETACH DELETE n")
    print("Database cleared")


def drop_all_indexes_and_constraints(graph):
    """Drop all indexes and constraints to start with a fresh db"""
    print("Dropping all existing indexes and constraints...")

    try:
        constraints = graph.query("SHOW CONSTRAINTS")
        for constraint in constraints:
            constraint_name = constraint.get("name")
            if constraint_name:
                graph.query(f"DROP CONSTRAINT {constraint_name} IF EXISTS")

        indexes = graph.query("SHOW INDEXES")
        for index in indexes:
            index_name = index.get("name")
            if index_name:
                graph.query(f"DROP INDEX {index_name} IF EXISTS")

        print("All indexes and constraints dropped")
    except Exception as e:
        print(f"Error dropping indexes and constraints: {e}")


def load_and_chunk_documents(repo_path, chunk_size=800, chunk_overlap=100):
    """Load documents from repository and split into chunks
    Using smaller chunk size to avoid token limit issues"""
    all_chunks = []
    print(f"Loading documents from {repo_path}...")
    for root, _, files in os.walk(repo_path):
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            try:
                if ext == ".py":
                    loader = PythonLoader(file_path)
                else:
                    loader = TextLoader(file_path)

                docs = loader.load()
                # Add metadata to each document
                for doc in docs:
                    doc.metadata["file_path"] = file_path
                    doc.metadata["language"] = str(
                        EXTENSION_TO_LANGUAGE.get(ext, "text")
                    )
                    doc.metadata["file_type"] = ext.lstrip(".")

                # Use language-specific splitter if available with smaller chunk size
                if ext in EXTENSION_TO_LANGUAGE:
                    splitter = RecursiveCharacterTextSplitter.from_language(
                        language=EXTENSION_TO_LANGUAGE[ext],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                else:  # fallback to generic splitter
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )

                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Processed a total of {len(all_chunks)} chunks from all documents")
    return all_chunks


def create_database_schema(graph):
    """Create database schema for storing code information"""
    print("Creating database schema...")
    # Each Document node type has a unique file path
    graph.query(
        """
    CREATE CONSTRAINT document_file_path IF NOT EXISTS
    FOR (d:Document) REQUIRE d.file_path IS UNIQUE
    """
    )

    # Each text chunk has a unique chunk ID
    graph.query(
        """
    CREATE CONSTRAINT text_chunk_id IF NOT EXISTS
    FOR (t:TextChunk) REQUIRE t.chunk_id IS UNIQUE
    """
    )

    # Create indexes for faster queries
    graph.query(
        "CREATE INDEX document_file_type IF NOT EXISTS FOR (d:Document) ON (d.file_type)"
    )
    graph.query(
        "CREATE INDEX document_language IF NOT EXISTS FOR (d:Document) ON (d.language)"
    )
    graph.query("CREATE INDEX feature_name IF NOT EXISTS FOR (f:Feature) ON (f.name)")
    graph.query(
        "CREATE INDEX component_name IF NOT EXISTS FOR (c:Component) ON (c.name)"
    )
    graph.query("CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)")
    graph.query("CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)")

    print("Database schema created")


def create_document_nodes(graph, chunks):
    """Create Document nodes from unique file paths"""
    print("Creating Document nodes...")

    # Creating document nodes:
    # 1. Process all chunks to extract unique file paths
    # 2. Store this info in dictionary with file_path key and languange and file_type as values

    file_paths = {}
    for chunk in chunks:
        file_path = chunk.metadata["file_path"]
        if file_path not in file_paths:
            file_paths[file_path] = {
                "language": chunk.metadata.get("language", "unknown"),
                "file_type": chunk.metadata.get("file_type", "unknown"),
            }

    # 3. For each entry in the dictionary, create a Document node with the file_path, language, and file_type properties
    for file_path, metadata in tqdm(file_paths.items(), desc="Creating Document nodes"):
        graph.query(
            """
        MERGE (d:Document {
            file_path: $file_path,
            language: $language,
            file_type: $file_type
        })
        """,
            {
                "file_path": file_path,
                "language": metadata["language"],
                "file_type": metadata["file_type"],
            },
        )

    print(f"Created {len(file_paths)} Document nodes")


def create_text_chunk_nodes(graph, chunks):
    """Create TextChunk nodes and connect to Documents"""
    print("Creating TextChunk nodes...")

    # Add unique IDs to chunks
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    # Create TextChunks node types and connect them to Document nodes by file_path
    for chunk in tqdm(chunks, desc="Creating TextChunk nodes"):
        graph.query(
            """
        MATCH (d:Document {file_path: $file_path})
        CREATE (t:TextChunk {
            chunk_id: $chunk_id,
            text: $text,
            start_idx: $start_idx,
            end_idx: $end_idx
        })
        CREATE (t)-[:PART_OF]->(d)
        """,
            {
                "file_path": chunk.metadata["file_path"],
                "chunk_id": chunk.metadata["chunk_id"],
                "text": chunk.page_content,
                "start_idx": chunk.metadata.get("start_index", 0),
                "end_idx": chunk.metadata.get("end_index", len(chunk.page_content)),
            },
        )

    print(f"Created {len(chunks)} TextChunk nodes")
    return chunks


def define_schema():
    """Define schema for LLMGraphTransformer with Function and Class nodes"""
    # Define allowed node types (including Function and Class)
    allowed_nodes = [
        "Feature",
        "Component",
        "Integration",
        "Function",
        "Class",
    ]
    allowed_relationships = [
        "IMPLEMENTS",
        "PROVIDES",
        "INTEGRATES_WITH",
        "DEPENDS_ON",
        "CALLS",
        "BELONGS_TO",
    ]

    node_properties = [
        "name",
        "description",
        "type",
        "signature",
        "chunk_id",
    ]

    return allowed_nodes, allowed_relationships, node_properties


def add_domain_guidance_prompt():
    """Create a domain-specific prompt for the LLMGraphTransformer with Function and Class extraction"""
    return """
    You are analyzing code from the CodeCarbon project, which tracks carbon emissions from computing.
    
    For each code chunk, identify:
    
    1. Features: Key functionality like emissions tracking, date range selection, or export capabilities
    2. Components: Main system components like EmissionsTracker or CarbonIntensityEstimator
    3. Integrations: External systems like cloud providers or ML frameworks
    4. Functions: Important functions that implement core logic
    5. Classes: Important classes that define components or features
    
    Each entity MUST have:
    - name: A short, descriptive name
    - description: A brief explanation of what it does
    - chunk_id: Copy the chunk_id value exactly from the metadata
    
    For Functions specifically, also include:
    - signature: The function signature (parameters and return type if available)
    
    Use these relationships:
    - IMPLEMENTS: Connect Features to their implementation
    - PROVIDES: Connect Components to Features they provide
    - INTEGRATES_WITH: Connect Components to external Integrations
    - DEPENDS_ON: Connect Components to other Components they depend on
    - CALLS: Connect Functions to other Functions they call
    - BELONGS_TO: Connect Functions to Classes they belong to
    
    FOCUS ONLY on the most important entities in the current chunk. Quality over quantity.
    """


def process_chunk_with_llm(llm_transformer, chunk, retries=3, backoff_factor=2):
    """Process a single chunk with error handling and backoff"""
    for attempt in range(retries):
        try:
            # Process the chunk
            graph_documents = llm_transformer.convert_to_graph_documents([chunk])
            return graph_documents
        except LengthFinishReasonError:
            # Fallback in case of token limit exceeded
            print(
                f"Token limit exceeded for chunk {chunk.metadata['chunk_id']}, creating basic representation"
            )
            return []
        # Basic strategy to deal with rate limits
        except Exception as e:
            if attempt < retries - 1:
                sleep_time = backoff_factor**attempt
                print(f"Error processing chunk: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"  Failed to process chunk after {retries} attempts: {e}")
                return []


def extract_entities_from_chunks(graph, chunks, batch_size=10):
    """Extract entities from chunks using LLMGraphTransformer"""
    print("Extracting entities from chunks...")

    # Define schema
    allowed_nodes, allowed_relationships, node_properties = define_schema()
    domain_guidance = add_domain_guidance_prompt()

    # Create LLM (using gpt-3.5-turbo for speed or gpt-4o-mini for better quality)
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-0125",
        api_key=os.getenv("OPEN_AI_API_KEY"),
    )

    # Create transformer
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        node_properties=node_properties,
        strict_mode=True,
        additional_instructions=domain_guidance,
    )

    # Process chunks in batches
    total_chunks = len(chunks)
    entity_counts = {
        "Feature": 0,
        "Component": 0,
        "Integration": 0,
        "Function": 0,
        "Class": 0,
    }
    relationship_counts = {
        "IMPLEMENTS": 0,
        "PROVIDES": 0,
        "INTEGRATES_WITH": 0,
        "DEPENDS_ON": 0,
        "CALLS": 0,
        "BELONGS_TO": 0,
    }

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch)} chunks)"
        )

        for chunk in tqdm(batch, desc=f"Batch {i//batch_size + 1}"):
            try:
                # Inside each bacth, process each chunk individually
                graph_documents = process_chunk_with_llm(llm_transformer, chunk)
                if not graph_documents:
                    continue

                # Extract entities and connect to TextChunk
                for doc in graph_documents:
                    for node in doc.nodes:
                        # Get chunk_id from the node or fall back to metadata
                        chunk_id = node.properties.get(
                            "chunk_id", chunk.metadata.get("chunk_id")
                        )

                        if not chunk_id:
                            continue

                        # Connect entity to chunk
                        try:
                            entity_type = node.type
                            entity_name = node.properties.get("name", "Unknown")

                            # Skip if entity name is missing or too generic
                            if not entity_name or entity_name == "Unknown":
                                continue

                            description = node.properties.get("description", "")

                            # For functions, include signature if available
                            additional_props = {}
                            if (
                                entity_type == "Function"
                                and "signature" in node.properties
                            ):
                                additional_props["signature"] = node.properties[
                                    "signature"
                                ]

                            # Prepare properties dict
                            props = {
                                "name": entity_name,
                                "description": description,
                                **additional_props,
                            }

                            # Build property string for MERGE query
                            props_strings = []
                            for key, value in props.items():
                                if value:
                                    props_strings.append(f"{key}: ${key}")

                            props_clause = ", ".join(props_strings)

                            # Create entity and connect to chunk
                            graph.query(
                                f"""
                            MATCH (t:TextChunk {{chunk_id: $chunk_id}})
                            MERGE (e:{entity_type} {{{props_clause}}})
                            MERGE (e)-[:DEFINED_IN]->(t)
                            """,
                                {"chunk_id": chunk_id, **props},
                            )

                            # Count entity types
                            if entity_type in entity_counts:
                                entity_counts[entity_type] += 1

                        except Exception as e:
                            print(f"  Error connecting entity to chunk: {e}")

                    # Create relationships between entities
                    for rel in doc.relationships:
                        try:
                            source_name = rel.source.properties.get("name")
                            target_name = rel.target.properties.get("name")
                            rel_type = rel.type

                            if source_name and target_name:
                                graph.query(
                                    f"""
                                MATCH (s:{rel.source.type} {{name: $source_name}})
                                MATCH (t:{rel.target.type} {{name: $target_name}})
                                MERGE (s)-[:{rel_type}]->(t)
                                """,
                                    {
                                        "source_name": source_name,
                                        "target_name": target_name,
                                    },
                                )

                                # Count relationship types
                                if rel_type in relationship_counts:
                                    relationship_counts[rel_type] += 1
                        except Exception as e:
                            print(f"  Error creating relationship: {e}")

            except Exception as e:
                print(f"  Error processing chunk {chunk.metadata.get('chunk_id')}: {e}")

    # Print entity counts
    print("\nEntity extraction complete.")
    print("Entities found:")
    for entity_type, count in entity_counts.items():
        print(f"  - {entity_type}: {count}")

    print("\nRelationships created:")
    for rel_type, count in relationship_counts.items():
        print(f"  - {rel_type}: {count}")


def main():
    """Main function to build the graph database"""
    # Connect to Neo4j
    url = os.getenv("NEO4J_URL")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    graph = Neo4jGraph(url=url, username=username, password=password)

    clear_database(graph)
    drop_all_indexes_and_constraints(graph)
    create_database_schema(graph)

    # Load and chunk documents
    chunks = load_and_chunk_documents(repo_path)
    # Create Document nodes
    create_document_nodes(graph, chunks)
    # Create TextChunk nodes
    chunks = create_text_chunk_nodes(graph, chunks)
    # Extract entities from chunks
    extract_entities_from_chunks(graph, chunks)

    print("\nFinal Graph Schema:")
    print(graph.get_schema)
    print("\nDatabase build complete!")


if __name__ == "__main__":
    main()


# Technical implementation note:
#
# Regarding the individual chunk processing approach:
# Each chunk is processed individually with (within a batch) withLLMGraphTransformer
# rather than chunking all documents together and passing them all at once to the transformer,
# Something like:
#               docs = loader.load()
#               chunks = splitter.split_documents(docs)
#               graph_documents = llm_transformer.convert_to_graph_documents(chunks)

# This has been done for some good reasons (after trial and error):
#
# 1. LLM Token Limits: The API call (OpenAI) would exceed token limits immediately. Each code chunk uses hundreds or thousands of tokens,
#    and LLMs have fixed context windows.
#
# 2. Granular Error Handling: If processing fails for one chunk, we don't lose progress on all chunks.
#    The try/except blocks (LengthFinishReasonError) show this design intent.
#
# 3. Retry Logic: The exponential backoff retry system (sleep_time = backoff_factor**attempt)
#    works at the individual chunk level, allowing precise retries (initial back off and #retries can be customized).
#
# 4. Incremental database updates: Each successfully processed chunk immediately writes
#    its entities to Neo4j, so we get partial results even if the process is interrupted (this was particularly useful in the development phase).
#
# 5. Progress tracking: Processing chunks individually allows us to show detailed progress
#    with tqdm, which it turns out be useful given the enormous amount of time needed to chunk all documents and process all chunks.
#
# I'm still learning about LLMs, but after some different stagies tried this approach maximizes reliability at the
# expense of some processing speed.
