
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
                MERGE (ref:ReferenceNode {id: -1, time: 0})
                SET ref.features = mean_features
            """)

            session.run("""
                CALL apoc.periodic.iterate(
                    "MATCH (n) WHERE n.id IS NOT NULL AND NOT (n:ReferenceNode) RETURN n",
                    "MATCH (ref:ReferenceNode {id: -1}) CREATE (n)<-[:CONNECTED_TO_REF]-(ref)",
                    {batchSize: 1000, parallel: false}
                )
            """)
    def add_edge_weights_depending_on_time_ratio(self):
        with self.driver.session() as session:
            session.run("""
                        MATCH (a)-[r]->(b)
CALL {
    WITH r, a, b
    WITH r, a.time AS ta, b.time AS tb
    WITH r,
        CASE
            WHEN ta = tb THEN 1.0
            WHEN ta < tb AND tb > 0 THEN toFloat(ta+1) / tb+1
            WHEN tb < ta AND ta > 0 THEN toFloat(tb+1) / ta+1
            ELSE 0.0
        END AS ratio_weight
    SET r.weight = ratio_weight
} IN TRANSACTIONS OF 1000 ROWS
                        """)
    def feature_neighborhood_aggregation_function(self):
        with self.driver.session() as session:
            session.run("""
CALL apoc.periodic.iterate(
    "MATCH (n) WHERE n.id > -1 RETURN n",
    "MATCH (n)<-[r]-(neighbor)
    WITH n, sum(coalesce(r.weight, 1.0)) AS total_weight, collect({features: neighbor.features, weight: coalesce(r.weight, 1.0)}) AS nb_data
    WITH n, [i IN range(0, size(n.features)-1) | 
        n.features[i] - (CASE WHEN total_weight > 0 THEN reduce(s = 0.0, nb IN nb_data | s + (nb.features[i] * nb.weight)) / total_weight ELSE 0.0 END)
    ] AS diff_features
    SET n.aggregated_features = diff_features",
    {batchSize: 1000, parallel: false}
)            """)
        pass
    ## TODO add pos encodings and difference scaled by time difference 
    ## TODO might make sense to also scale and add pos encodings? Idk yet but might further push AUROC up
