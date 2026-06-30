
from neo4j import GraphDatabase
import torch
import numpy as np
from torch_geometric.data import Data

def get_subgraph_query(split='train'):
    split = split.upper()
    return f"""
UNWIND $seed_ids AS seedId
MATCH (seed:{split})
WHERE seed.id = seedId AND seed.id > -1

CALL(seed){{
    MATCH (seed:{split})<--(n1:{split})
    WITH n1
    ORDER BY n1.id
    LIMIT $limit_1
    RETURN collect(n1) AS hop1_nodes
}}

CALL(hop1_nodes) {{
    UNWIND hop1_nodes AS h1
    MATCH (h1:{split})<--(h2:{split})
    WITH h2
    ORDER BY h2.id
    LIMIT $limit_2
    RETURN collect(DISTINCT h2) AS hop2_nodes
}}

WITH seed, hop1_nodes, hop2_nodes, (hop1_nodes + hop2_nodes + [seed]) AS all_sampled_nodes
UNWIND all_sampled_nodes AS n
WITH DISTINCT n AS sampledNode, all_sampled_nodes

MATCH (sampledNode:{split})-[r]->(target:{split})
WHERE target IN all_sampled_nodes AND sampledNode.id > -1 AND target.id > -1

WITH collect(DISTINCT sampledNode) AS uniqueNodes, collect(DISTINCT r) AS uniqueEdges
RETURN 
    [n IN uniqueNodes | n.id] AS node_ids,
    [n IN uniqueNodes | n.features_scaled] AS features,
    [n IN uniqueNodes | toInteger(n.label)] AS labels,
    [e IN uniqueEdges | [startNode(e).id, endNode(e).id, coalesce(e.weight,1.0)]] AS edge_data
    """
def get_full_graph_query(split='train'):
    """
    Retrieves all nodes and edges for a full-batch approach.
    Replaces the multi-hop sampler with a global MATCH.
    """
    split = split.upper()
    return f"""
    // 1. Match all valid nodes in the requested split
    MATCH (n:{split})
    WHERE n.id > -1
    
    // 2. Optionally match all intra-split relationships
    // OPTIONAL MATCH ensures we still retrieve isolated nodes
    OPTIONAL MATCH (n)-[r]->(m:{split})
    WHERE m.id > -1
    
    // 3. Aggregate nodes and edges into distinct collections
    WITH 
        collect(DISTINCT n) AS uniqueNodes, 
        collect(DISTINCT r) AS uniqueEdges
        
    // 4. Format and return lists to exactly match the PyG extraction logic
    RETURN 
        [node IN uniqueNodes | node.id] AS node_ids,
        [node IN uniqueNodes | node.features_scaled] AS features,
        [node IN uniqueNodes | toInteger(node.label)] AS labels,
        [edge IN uniqueEdges WHERE edge IS NOT NULL | [startNode(edge).id, endNode(edge).id, coalesce(edge.weight, 1.0)]] AS edge_data
    """
    

class Neo4jConnector:
    def __init__(self, uri="bolt://host.docker.internal:7688", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def get_patient_ids(self, split_name):
        with self.driver.session() as session:
            seed_ids = session.run(f"MATCH (n:{split_name}) RETURN COLLECT(DISTINCT n.patientId) as ids").single().get("ids")
        return seed_ids
    
    def get_sbc_patient_train_ids(self):
        return self.get_patient_ids('SBC_TRAIN')
    
    def get_sbc_patient_test_ids(self):
        return self.get_patient_ids('SBC_TEST')

    def get_sbc_patient_ext_test_ids(self):
        return self.get_patient_ids('SBC_EXT_TEST')

    def get_mimic_patient_train_ids(self):
        return self.get_patient_ids('MIMIC_TRAIN')

    def get_mimic_patient_test_ids(self):
        return self.get_patient_ids('MIMIC_TEST')

    def get_mimic_patient_val_ids(self):
        return self.get_patient_ids('MIMIC_VAL')

    def get_ids_from_patient_ids(self, patient_ids, split_name):
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (n:{split_name})
                WHERE n.patientId IN $patient_ids
                RETURN COLLECT(n.id) AS ids
            """, patient_ids=patient_ids).single()
            return result["ids"] if result else []

    def get_sbc_train_ids(self, patient_ids):
        return self.get_ids_from_patient_ids(patient_ids, 'SBC_TRAIN')

    def get_sbc_test_ids(self, patient_ids):
        return self.get_ids_from_patient_ids(patient_ids, 'SBC_TEST')

    def get_sbc_ext_test_ids(self, patient_ids):
        return self.get_ids_from_patient_ids(patient_ids, 'SBC_EXT_TEST')

    def get_mimic_train_ids(self, patient_ids):
        return self.get_ids_from_patient_ids(patient_ids, 'MIMIC_TRAIN')

    def get_mimic_test_ids(self, patient_ids):
        return self.get_ids_from_patient_ids(patient_ids, 'MIMIC_TEST')

    def get_mimic_val_ids(self, patient_ids):
        return self.get_ids_from_patient_ids(patient_ids, 'MIMIC_VAL')

    def get_seeded_subgraphs(self, batch_seeds, hops_limits, split_name):
        with self.driver.session() as session:
            result = session.run(
                get_subgraph_query(split=split_name), 
                seed_ids=batch_seeds, 
                limit_1=hops_limits[0], 
                limit_2=hops_limits[1]
            )
            record = result.single()

        edge_np = np.array(record['edge_data'], dtype=np.float64)
        if not record or not record['node_ids'] or edge_np.size == 0: 
            return Data() 

        global_src = torch.from_numpy(edge_np[:, 0].astype(np.int64))
        global_dst = torch.from_numpy(edge_np[:, 1].astype(np.int64))
        weights    = torch.from_numpy(edge_np[:, 2].astype(np.float32)).unsqueeze(1)

        all_node_ids = torch.tensor(record['node_ids'], dtype=torch.long)
        x = torch.tensor(record['features'], dtype=torch.float)
        y = torch.tensor(record['labels'], dtype=torch.float).unsqueeze(1)

        sorted_ids, sort_idx = torch.sort(all_node_ids)
        x = x[sort_idx]
        y = y[sort_idx]

        src_local = torch.searchsorted(sorted_ids, global_src)
        dst_local = torch.searchsorted(sorted_ids, global_dst)
        edge_index = torch.stack([src_local, dst_local], dim=0)
        batch_seeds_tensor = torch.tensor(batch_seeds, dtype=torch.long)
        batch_mask = torch.isin(sorted_ids, batch_seeds_tensor)        

        return Data(x=x, edge_index=edge_index, edge_attr=weights, y=y, batch_mask=batch_mask)
    
    import numpy as np
import torch
from torch_geometric.data import Data

def get_full_graph(self):
    """
    Retrieves the entire graph for full-batch GNN training.
    """
    with self.driver.session() as session:
        result = session.run(get_full_graph_query())
        record = result.single()

    if not record or not record.get('node_ids'): 
        return Data() 
    all_node_ids = torch.tensor(record['node_ids'], dtype=torch.long)
    x = torch.tensor(record['features'], dtype=torch.float)
    y = torch.tensor(record['labels'], dtype=torch.float).unsqueeze(1)
    
    sorted_ids, sort_idx = torch.sort(all_node_ids)
    x = x[sort_idx]
    y = y[sort_idx]
    edge_np = np.array(record.get('edge_data', []), dtype=np.float64)
    
    if edge_np.size > 0:
        global_src = torch.from_numpy(edge_np[:, 0].astype(np.int64))
        global_dst = torch.from_numpy(edge_np[:, 1].astype(np.int64))
        weights    = torch.from_numpy(edge_np[:, 2].astype(np.float32)).unsqueeze(1)

        # Map global Neo4j IDs to local PyG indices (0 to N-1)
        src_local = torch.searchsorted(sorted_ids, global_src)
        dst_local = torch.searchsorted(sorted_ids, global_dst)
        edge_index = torch.stack([src_local, dst_local], dim=0)
    else:
        # Handle edge case where the graph has nodes but no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        weights = torch.empty((0, 1), dtype=torch.float32)

    return Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=weights, 
        y=y,
        batch_mask=torch.ones(x.size(0), dtype=torch.bool) 
    )

    def scale_and_add_positional_encodings(self, train_split_name, *test_split_names):
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (n:{train_split_name})
                UNWIND range(0, size(n.features)-1) AS i
                WITH i, min(n.features[i]) AS global_min, max(n.features[i]) AS global_max
                ORDER BY i
                RETURN collect(global_min) AS mins, collect(global_max) AS maxs
            """).single()
            
            mins = result["mins"]
            maxs = result["maxs"]
            print("Feature mins:", mins, "maxs:", maxs)

            scale_query = """
                WHERE n.id > -1 AND  n.features_scaled IS NULL
                CALL (n) {
                    SET n.features_scaled = [i IN range(0, size(n.features)-1) | 
                        CASE 
                            WHEN ($maxs[i] - $mins[i]) = 0 THEN 0.0 + n.pos_encodings[i]
                            ELSE (n.features[i] - $mins[i]) / ($maxs[i] - $mins[i]) + n.pos_encodings[i]
                        END
                    ]
                } IN TRANSACTIONS OF 2000 ROWS
            """

            session.run(f"MATCH (n:{train_split_name}) {scale_query}", mins=mins, maxs=maxs)

            for split_name in test_split_names:
                session.run(f"MATCH (n:{split_name}) {scale_query}", mins=mins, maxs=maxs)
            print("Scaled features and added positional encodings.")

    def has_sbc_nodes(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n:SBC_TRAIN) RETURN COUNT(n) AS count").single()
            return result["count"] > 0      

    def has_mimic_nodes(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n:MIMIC_TRAIN) RETURN COUNT(n) AS count").single()
            return result["count"] > 0
        
    def get_node_count(self, split_name):
        with self.driver.session() as session:
            result = session.run(f"MATCH (n:{split_name}) RETURN COUNT(n) AS count").single()
            return result["count"] if result else 0