from neo4j import GraphDatabase
from Neo4jQueries import Neo4jQueries
import os

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
        mimic_split_dict = {
            "train": "MIMIC_TRAIN",
            "test": "MIMIC_TEST",
            "val": "MIMIC_VAL"
        }
        print("Cleared existing data in the database.")
        for split in ["train", "val", "test"]:      
            neo4j_queries = Neo4jQueries(mimic_split_dict[split])
            print(f"Attempting to upload {split} data to Neo4j database...")      
            
            try:
                session.run(neo4j_queries.get_nodes_creation_query(), file=f"file:///mimic_{split}_nodes.csv")  
                print(f"Nodes for {split} uploaded.")     
                
                session.run(neo4j_queries.get_node_id_index_query())     
                session.run("CALL db.awaitIndexes()")
                
                session.run(neo4j_queries.get_edges_creation_query(), file=f"file:///mimic_{split}_edges.csv")  
                print(f"Edges for {split} uploaded.")          
                
                session.run(neo4j_queries.get_pos_enc_creation_query(), file=f"file:///mimic_{split}_pos_encodings.csv")
                print(f"Postional encodings for {split} uploaded.")
                
            except Exception as e:
                print(f"Skipping {split} split. A file was missing or invalid. Error details: {e}")
                continue
        
        split_dict = {
            "": "SBC_TRAIN",
            "_validation": "SBC_TEST",
            "_ext_validation": "SBC_EXT_TEST"
        }
        for split in ["", "_validation", "_ext_validation"]:      
            neo4j_queries = Neo4jQueries(split_dict[split])
            print(f"Attempting to upload sbc{split} data to Neo4j database...")      
            
            try:
                session.run(neo4j_queries.get_nodes_creation_query(), file=f"file:///sbc{split}_nodes.csv")  
                print(f"Nodes for sbc{split} uploaded.")     
                
                session.run(neo4j_queries.get_node_id_index_query())     
                session.run("CALL db.awaitIndexes()")
                
                session.run(neo4j_queries.get_edges_creation_query(), file=f"file:///sbc{split}_edges.csv")  
                print(f"Edges for sbc{split} uploaded.")          
                
                session.run(neo4j_queries.get_pos_enc_creation_query(), file=f"file:///sbc{split}_pos_encodings.csv")
                print(f"Postional encodings for sbc{split} uploaded.")

                # session.run(neo4j_queries.get_self_loops_creation_query())
                # print(f"Self-loops for sbc{split} created.")
                
            except Exception as e:
                print(f"Skipping sbc{split} split. Error details: {e}")
                continue
        
    neo4j_driver.close()