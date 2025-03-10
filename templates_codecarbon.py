CYPHER_EXAMPLES = [
    {
        "question": "Can the user select a time range to analyze the energy emissions?",
        "query": """
        // Target specific files directly for this feature
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/repository_emissions.py' 
           OR d.file_path CONTAINS '/date-range-picker.tsx'
           OR d.file_path CONTAINS '/components.py'
           OR d.file_path CONTAINS '/emissions_tracker.py'
           OR d.file_path CONTAINS '/experiments.ts'
           OR d.file_path CONTAINS '/page.tsx'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'date' OR tc.text CONTAINS 'time' OR tc.text CONTAINS 'range' OR tc.text CONTAINS 'emissions'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "Does CodeCarbon support different cloud providers for energy tracking?",
        "query": """
        // Target exact files related to cloud providers
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/cloud.py' 
           OR d.file_path CONTAINS '/emissions.py'
           OR d.file_path CONTAINS '/emissions_tracker.py'
           OR d.file_path CONTAINS '/methodology.html'
           OR d.file_path CONTAINS '/faq.html'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'cloud' 
           OR tc.text CONTAINS 'provider' 
           OR tc.text CONTAINS 'AWS' 
           OR tc.text CONTAINS 'Azure' 
           OR tc.text CONTAINS 'GCP'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "How can the user export energy consumption reports?",
        "query": """
        // Target files related to exporting and reporting
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/emissions_tracker.py' 
           OR d.file_path CONTAINS '/output.py'
           OR d.file_path CONTAINS '/runs.ts'
           OR d.file_path CONTAINS '/organizations.ts'
           OR d.file_path CONTAINS '/experiments.ts'
           OR d.file_path CONTAINS '/run-report.ts'
           OR d.file_path CONTAINS '/experiment-report.ts'
           OR d.file_path CONTAINS '/organization-report.ts'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'export' 
           OR tc.text CONTAINS 'report' 
           OR tc.text CONTAINS 'csv' 
           OR tc.text CONTAINS 'output'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "Is it possible to track energy usage for multiple processes or threads?",
        "query": """
        // Target files handling multiple processes or threads
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/emissions_tracker.py' 
           OR d.file_path CONTAINS '/resource_tracker.py'
           OR d.file_path CONTAINS '/methodology.html'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'process' 
           OR tc.text CONTAINS 'thread' 
           OR tc.text CONTAINS 'parallel' 
           OR tc.text CONTAINS 'multi'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "What are the different ways CodeCarbon estimates energy consumption?",
        "query": """
        // Target files handling energy estimation methods
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/hardware.py' 
           OR d.file_path CONTAINS '/cpu.py'
           OR d.file_path CONTAINS '/powermetrics.py'
           OR d.file_path CONTAINS '/gpu.py'
           OR d.file_path CONTAINS '/co2_signal.py'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'estimat' 
           OR tc.text CONTAINS 'energy' 
           OR tc.text CONTAINS 'consumption'
           OR tc.text CONTAINS 'measur'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "How does CodeCarbon handle missing hardware energy data?",
        "query": """
        // Target files handling missing hardware data
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/hardware.py' 
           OR d.file_path CONTAINS '/measure.py'
           OR d.file_path CONTAINS '/emissions_tracker.py'
           OR d.file_path CONTAINS '/powermetrics.py'
           OR d.file_path CONTAINS '/cpu.py'
           OR d.file_path CONTAINS '/gpu.py'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'missing' 
           OR tc.text CONTAINS 'default' 
           OR tc.text CONTAINS 'fallback'
           OR tc.text CONTAINS 'estimate'
           OR tc.text CONTAINS 'unavailable'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "Can the user specify a custom carbon intensity factor for calculations?",
        "query": """
        // Target files related to custom carbon intensity
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/emissions_tracker.py' 
           OR d.file_path CONTAINS '/emissions.py'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'custom' 
           OR tc.text CONTAINS 'intensity' 
           OR tc.text CONTAINS 'factor'
           OR tc.text CONTAINS 'carbon'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "How does CodeCarbon determine the geographic location for carbon intensity calculations?",
        "query": """
        // Target files handling geographic location
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/geography.py' 
           OR d.file_path CONTAINS '/cloud.py'
           OR d.file_path CONTAINS '/emissions_tracker.py'
           OR d.file_path CONTAINS '/emissions.py'
           OR d.file_path CONTAINS '/co2_signal.py'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'geographic' 
           OR tc.text CONTAINS 'location' 
           OR tc.text CONTAINS 'region'
           OR tc.text CONTAINS 'country'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "How can I integrate CodeCarbon with machine learning experiments in TensorFlow or PyTorch?",
        "query": """
        // Target files about ML framework integration
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/mnist-sklearn.py' 
           OR d.file_path CONTAINS '/bert_inference.py'
           OR d.file_path CONTAINS '/pytorch-multigpu-example.py'
           OR d.file_path CONTAINS '/mnist_callback.py'
           OR d.file_path CONTAINS '/mnist_decorator.py'
           OR d.file_path CONTAINS '/mnist.py'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'tensorflow' 
           OR tc.text CONTAINS 'pytorch' 
           OR tc.text CONTAINS 'model'
           OR tc.text CONTAINS 'train'
           OR tc.text CONTAINS 'callback'
           OR tc.text CONTAINS 'decorator'
        RETURN d.file_path, tc.text
        """,
    },
    {
        "question": "Is there a way to log energy consumption data into a database instead of CSV files?",
        "query": """
        // Target files related to database logging
        MATCH (d:Document)
        WHERE d.file_path CONTAINS '/output.py' 
           OR d.file_path CONTAINS '/logger.py'
           OR d.file_path CONTAINS '/database_manager.py'
           OR d.file_path CONTAINS '/database.py'
           OR d.file_path CONTAINS '/sql_models.py'
           OR d.file_path CONTAINS '/env.py'
        MATCH (tc:TextChunk)-[:PART_OF]->(d)
        WHERE tc.text CONTAINS 'database' 
           OR tc.text CONTAINS 'sql' 
           OR tc.text CONTAINS 'log'
           OR tc.text CONTAINS 'store'
           OR tc.text CONTAINS 'record'
        RETURN d.file_path, tc.text
        """,
    },
]

PREFIX = """You are a Neo4j expert for searching the CodeCarbon codebase. Create Cypher queries to find specific code files and their relevant text content that address the user's question.
            
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

QA_TEMPLATE = """You are an expert on the CodeCarbon codebase that helps developers understand how the library works.
Based on the retrieved code snippets from specific files, answer the user's question about CodeCarbon functionality.

Code context from relevant files:
{context}

User question: {question}

Your answer must:
1. Be direct and focused on CodeCarbon functionality
2. Include specific implementation details from the retrieved files
3. Reference the exact file names where appropriate (e.g., "In cloud.py, CodeCarbon implements...")
4. Focus on how the feature is implemented or can be used
5. Be limited to 5 sentences maximum

If the retrieved code snippets don't provide enough information, clearly state what is known and what additional information would be needed.

Answer:"""
