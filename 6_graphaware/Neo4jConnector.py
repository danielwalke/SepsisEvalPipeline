
from neo4j import GraphDatabase
import torch
import numpy as np
from torch_geometric.data import Data

class Neo4jConnector:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def get_ids(self, split_name):
        with self.driver.session() as session:
            seed_ids = session.run(f"MATCH (n:{split_name}) RETURN COLLECT(n.id) as ids").single().get("ids")
        return seed_ids
    
    def get_train_ids(self):
        return self.get_ids('SBC_TRAIN')
    
    def get_test_ids(self):
        return self.get_ids('SBC_TEST')
    
    def append_ref_node(self):
        with self.driver.session() as session:
            session.run("""
                MATCH (n)
                WHERE n.id IS NOT NULL AND n.features IS NOT NULL
                UNWIND range(0, size(n.features)-1) AS i
                WITH i, avg(n.features[i]) AS val
                ORDER BY i
                WITH collect(val) AS mean_features
                MERGE (ref:ReferenceNode {id: -1})
                SET ref.features = mean_features
            """)

            session.run("""
                CALL apoc.periodic.iterate(
                    "MATCH (n) WHERE n.id IS NOT NULL AND NOT (n:ReferenceNode) RETURN n",
                    "MATCH (ref:ReferenceNode {id: -1}) CREATE (n)<-[:CONNECTED_TO_REF]-(ref)",
                    {batchSize: 1000, parallel: false}
                )
            """)
    
    def feature_neighborhood_aggregation_function(self):
        with self.driver.session() as session:
            session.run("""
                CALL apoc.periodic.iterate(
                "MATCH (n) WHERE n.id > -1 RETURN n",
                "MATCH (n)<--(neighbor)
                WITH n, collect(neighbor.features) AS nb_features
                WITH n, [i IN range(0, size(n.features)-1) | n.features[i] - apoc.coll.avg([f IN nb_features | f[i]])] AS diff_features
                SET n.aggregated_features = diff_features",
                {batchSize: 1000, parallel: false}
                )
            """)
        pass
    
    ## TODO might make sense to also scale and add pos encodings? Idk yet but might further push AUROC up