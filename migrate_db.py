import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
import time

# Load environment variables
load_dotenv()


def connect_to_db():
    """Connect to the Neo4j database"""
    url = os.getenv("NEO4J_URL")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    # Connect to your actual database name
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    return Neo4jGraph(url=url, username=username, password=password, database=database)


def create_missing_relationships(graph):
    """Create the missing relationships needed for your queries without rebuilding the database"""
    print("Creating missing relationships...")

    # Track counts for monitoring
    created_counts = {
        "IMPLEMENTED_IN": 0,
        "PROVIDES": 0,
        "INTEGRATES_WITH": 0,
        "CALLS": 0,
        "BELONGS_TO": 0,
    }

    # 1. Create IMPLEMENTED_IN relationships: Feature -> Document
    print("Creating Feature-IMPLEMENTED_IN->Document relationships...")
    result = graph.query(
        """
    MATCH (f:Feature)-[:DEFINED_IN]->(tc:TextChunk)-[:PART_OF]->(d:Document)
    WHERE NOT EXISTS((f)-[:IMPLEMENTED_IN]->(d))
    WITH f, d
    MERGE (f)-[:IMPLEMENTED_IN]->(d)
    RETURN count(*) as count
    """
    )
    created_counts["IMPLEMENTED_IN"] = result[0]["count"]
    print(f"  Created {created_counts['IMPLEMENTED_IN']} IMPLEMENTED_IN relationships")

    # 2. Create PROVIDES relationships using text-based inference
    # Look for Component and Feature nodes defined in the same or related TextChunks
    print("Creating Component-PROVIDES->Feature relationships...")
    result = graph.query(
        """
    MATCH (c:Component)-[:DEFINED_IN]->(tc1:TextChunk)
    MATCH (f:Feature)-[:DEFINED_IN]->(tc2:TextChunk)
    WHERE tc1.chunk_id = tc2.chunk_id OR 
          tc1.text CONTAINS f.name OR 
          tc2.text CONTAINS c.name
    WITH c, f
    WHERE NOT EXISTS((c)-[:PROVIDES]->(f))
    MERGE (c)-[:PROVIDES]->(f)
    RETURN count(*) as count
    """
    )
    created_counts["PROVIDES"] = result[0]["count"]
    print(f"  Created {created_counts['PROVIDES']} PROVIDES relationships")

    # 3. Create INTEGRATES_WITH relationships for Integrations
    print("Creating Integration-INTEGRATES_WITH->Document relationships...")
    result = graph.query(
        """
    MATCH (i:Integration)-[:DEFINED_IN]->(tc:TextChunk)-[:PART_OF]->(d:Document)
    WHERE NOT EXISTS((i)-[:INTEGRATES_WITH]->(d))
    MERGE (i)-[:INTEGRATES_WITH]->(d)
    RETURN count(*) as count
    """
    )
    created_counts["INTEGRATES_WITH"] = result[0]["count"]
    print(
        f"  Created {created_counts['INTEGRATES_WITH']} INTEGRATES_WITH relationships"
    )

    # 4. Create CALLS relationships between Functions
    # This is more complex - we'll use text analysis to infer function calls
    print("Creating Function-CALLS->Function relationships...")
    result = graph.query(
        """
    MATCH (fn1:Function)
    MATCH (fn2:Function)
    WHERE fn1 <> fn2
    WITH fn1, fn2
    MATCH (fn1)-[:DEFINED_IN]->(tc:TextChunk)
    WHERE tc.text CONTAINS fn2.name
    AND NOT EXISTS((fn1)-[:CALLS]->(fn2))
    MERGE (fn1)-[:CALLS]->(fn2)
    RETURN count(*) as count
    """
    )
    created_counts["CALLS"] = result[0]["count"]
    print(f"  Created {created_counts['CALLS']} CALLS relationships")

    # 5. Create BELONGS_TO relationships between Functions and Classes
    print("Creating Function-BELONGS_TO->Class relationships...")
    result = graph.query(
        """
    MATCH (fn:Function)-[:DEFINED_IN]->(tc1:TextChunk)
    MATCH (c:Class)-[:DEFINED_IN]->(tc2:TextChunk)
    MATCH (tc1)-[:PART_OF]->(d:Document)<-[:PART_OF]-(tc2)
    WHERE tc1.chunk_id = tc2.chunk_id OR d IS NOT NULL
    WITH fn, c
    WHERE NOT EXISTS((fn)-[:BELONGS_TO]->(c))
    MERGE (fn)-[:BELONGS_TO]->(c)
    RETURN count(*) as count
    """
    )
    created_counts["BELONGS_TO"] = result[0]["count"]
    print(f"  Created {created_counts['BELONGS_TO']} BELONGS_TO relationships")

    return created_counts


def validate_relationships(graph):
    """Validate that the relationships needed for queries exist"""
    print("\nValidating relationships...")

    # Check counts of each relationship type
    result = graph.query(
        """
    MATCH ()-[r]->()
    RETURN type(r) as type, count(r) as count
    ORDER BY count DESC
    """
    )

    print("Relationship counts in database:")
    for row in result:
        print(f"  {row['type']}: {row['count']}")

    # Verify specific relationships used in your queries
    relationship_checks = [
        "MATCH (f:Feature)-[:IMPLEMENTED_IN]->(d:Document) RETURN count(*) as count",
        "MATCH (c:Component)-[:PROVIDES]->(f:Feature) RETURN count(*) as count",
        "MATCH (i:Integration)-[:INTEGRATES_WITH]->() RETURN count(*) as count",
        "MATCH (fn:Function)-[:DEFINED_IN]->() RETURN count(*) as count",
    ]

    for check in relationship_checks:
        count = graph.query(check)[0]["count"]
        print(f"  {check}: {count}")


def update_query_engine(graph):
    """Test the query engine with the updated relationships"""
    print("\nTesting query against updated database...")
    # Perform a simple test query using the relationships you created
    test_query = """
    MATCH (f:Feature)-[:IMPLEMENTED_IN]->(d:Document)
    WHERE f.name CONTAINS 'track' OR f.name CONTAINS 'monitor' 
    RETURN f.name, d.file_path LIMIT 5
    """

    result = graph.query(test_query)
    print(f"Found {len(result)} results for test query")
    for row in result:
        print(f"  Feature: {row.get('f.name')}, Document: {row.get('d.file_path')}")


def main():
    start_time = time.time()

    # Connect to Neo4j
    graph = connect_to_db()

    # Create missing relationships
    created_counts = create_missing_relationships(graph)

    # Validate the relationships
    validate_relationships(graph)

    # Test the query engine
    update_query_engine(graph)

    end_time = time.time()
    print(f"\nMigration completed in {end_time - start_time:.2f} seconds")
    print("Summary of created relationships:")
    for rel_type, count in created_counts.items():
        print(f"  {rel_type}: {count}")

    print("\nYou can now use the updated database with your existing query_engine.py")


if __name__ == "__main__":
    main()
