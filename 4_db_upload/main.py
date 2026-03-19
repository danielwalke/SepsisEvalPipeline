from neo4j import GraphDatabase
from Neo4jQueries import Neo4jQueries

if __name__ == "__main__":
    uri = "bolt://neo4j_db:7687"
    user = "neo4j"
    password = "password"
    print(uri, user, password)
    neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with neo4j_driver.session() as session:
        # Batch delete to prevent OutOfMemory errors
        session.run("""
            MATCH (n)
            CALL {
                WITH n
                DETACH DELETE n
            } IN TRANSACTIONS OF 500 ROWS
        """)
        print(session.run("MATCH (n) RETURN count(n) AS count").single().get("count"))
        
    with neo4j_driver.session() as session:
        print("Cleared existing data in the database.")
        # for split in ["train", "val", "test"]:      
        #     neo4j_queries = Neo4jQueries(split.upper())
        #     print(f"Uploading {split} data to Neo4j database...")      
        #     print(neo4j_queries.get_nodes_creation_query())
        #     session.run(neo4j_queries.get_nodes_creation_query(), file=f"file:///mimic_{split}_nodes.csv")  
        #     print(f"Nodes for {split} uploaded.")     
        #     session.run(neo4j_queries.get_node_id_index_query())     
        #     session.run("CALL db.awaitIndexes()")
        #     session.run(neo4j_queries.get_edges_creation_query(), file=f"file:///mimic_{split}_edges.csv")  
        #     print(f"Edges for {split} uploaded.")          
        #     session.run(neo4j_queries.get_pos_enc_creation_query(), file=f"file:///mimic_{split}_pos_encodings.csv")
        #     print(f"Postional encodings for {split} uploaded.")
        
        split_dict = {
            "": "SBC_TRAIN",
            "_validation": "SBC_TEST",
            "_ext_validation": "SBC_EXT_TEST"
        }
        for split in ["", "_validation", "_ext_validation"]:      
            neo4j_queries = Neo4jQueries(split_dict[split])
            print(f"Uploading {split} data to Neo4j database...")      
            print(neo4j_queries.get_nodes_creation_query())
            print("file:///sbc{split}_nodes.csv")
            session.run(neo4j_queries.get_nodes_creation_query(), file=f"file:///sbc{split}_nodes.csv")  
            print(f"Nodes for {split} uploaded.")     
            session.run(neo4j_queries.get_node_id_index_query())     
            session.run("CALL db.awaitIndexes()")
            session.run(neo4j_queries.get_edges_creation_query(), file=f"file:///sbc{split}_edges.csv")  
            print(f"Edges for {split} uploaded.")          
            session.run(neo4j_queries.get_pos_enc_creation_query(), file=f"file:///sbc{split}_pos_encodings.csv")
            print(f"Postional encodings for {split} uploaded.")
        
    neo4j_driver.close()