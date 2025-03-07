CYPHER_EXAMPLES = [
    {
        "question": "Where is the main functionality implemented?",
        "query": """
        // Target specific files for core functionality
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/core/' 
           OR d.file_path CONTAINS '/main.py'
           OR d.file_path CONTAINS '/api.py'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'function' OR tc.text CONTAINS 'class' OR tc.text CONTAINS 'main'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "How does the project handle external integrations?",
        "query": """
        // Target exact files related to external integrations
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/integrations/' 
           OR d.file_path CONTAINS '/connectors/'
           OR d.file_path CONTAINS '/external_apis.py'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'connect' 
           OR tc.text CONTAINS 'api' 
           OR tc.text CONTAINS 'external'
           OR tc.text CONTAINS 'integration'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "Can users export data from the application?",
        "query": """
        // Target files related to exporting and reporting
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/export.py' 
           OR d.file_path CONTAINS '/output.py'
           OR d.file_path CONTAINS '/reporting/'
           OR d.file_path CONTAINS '/data_output/'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'export' 
           OR tc.text CONTAINS 'report' 
           OR tc.text CONTAINS 'csv' 
           OR tc.text CONTAINS 'output'
           OR tc.text CONTAINS 'save'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "How does the project handle multithreading or concurrency?",
        "query": """
        // Target files handling multiple processes or threads
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/threading.py' 
           OR d.file_path CONTAINS '/concurrency.py'
           OR d.file_path CONTAINS '/async/'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'process' 
           OR tc.text CONTAINS 'thread' 
           OR tc.text CONTAINS 'parallel' 
           OR tc.text CONTAINS 'multi'
           OR tc.text CONTAINS 'async'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "What configuration options are available?",
        "query": """
        // Target files handling configuration
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/config.py' 
           OR d.file_path CONTAINS '/settings.py'
           OR d.file_path CONTAINS '/options.py'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'config' 
           OR tc.text CONTAINS 'setting' 
           OR tc.text CONTAINS 'option'
           OR tc.text CONTAINS 'parameter'
        RETURN d.file_path, tc.text
        """,
    },
]

PREFIX = """You are a Neo4j expert for searching a codebase. Create Cypher queries to find specific code files and their relevant text content that address the user's question.
            
            IMPORTANT SCHEMA INFORMATION:
            - Document nodes have file_path but NO text property
            - TextChunk nodes store the actual text content and are connected to Documents
            - Always include TextChunk nodes in your query to access text content
            
            IMPORTANT QUERY GUIDELINES:
            - Target the SPECIFIC FILES that are most likely to answer the question
            - For each question, identify which files would likely contain the answer
            - Use "d.file_path CONTAINS '/filename.py'" to target specific files
            - Always match both Document and TextChunk: MATCH (tc:TextChunk)-[:PART_OF]->(d:Document)
            - Use relevant keywords in the TextChunk content filter
            - Return both d.file_path and tc.text
            - Do NOT use LIMIT as we want all relevant text chunks from the targeted files
            
            Database schema: {schema}
            
            Here are examples of effective Cypher queries that target specific files:"""

QA_TEMPLATE = """You are an expert on the codebase that helps developers understand how the project works.
Based on the retrieved code snippets from specific files, answer the user's question about the project functionality.

Code context from relevant files:
{context}

User question: {question}

Your answer must:
1. Be direct and focused on the project functionality
2. Include specific implementation details from the retrieved files
3. Reference the exact file names where appropriate (e.g., "In config.py, the project implements...")
4. Focus on how the feature is implemented or can be used
5. Be limited to 5 sentences maximum

If the retrieved code snippets don't provide enough information, clearly state what is known and what additional information would be needed.

Answer:"""
