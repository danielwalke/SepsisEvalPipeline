
from neo4j import GraphDatabase
import numpy as np

class Neo4jConnector:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()

    def get_patient_ids(self, split_name):
        with self.driver.session() as session:
            result = session.run(f"MATCH (n:{split_name}) RETURN COLLECT(DISTINCT n.patientId) as ids")
            return result.single().get("ids")

    def get_ids(self, split_name):
        with self.driver.session() as session:
            result = session.run(f"MATCH (n:{split_name}) RETURN COLLECT(id(n)) as ids")
            return result.single().get("ids")

    def get_sbc_train_ids(self, train_patient_ids):
        with self.driver.session() as session:
            result = session.run("MATCH (n:SBC_TRAIN) WHERE n.patientId IN $ids RETURN COLLECT(id(n)) as ids", ids=train_patient_ids)
            return result.single().get("ids")

    def get_sbc_val_ids(self, val_patient_ids):
        with self.driver.session() as session:
            result = session.run("MATCH (n:SBC_TRAIN) WHERE n.patientId IN $ids RETURN COLLECT(id(n)) as ids", ids=val_patient_ids)
            return result.single().get("ids")

    def fetch_data_batch(self, node_label, condition, skip, limit, framework, target_ids=None):
        with self.driver.session() as session:
            return session.execute_read(
                self._fetch_data_batch_tx, 
                node_label, 
                condition, 
                skip, 
                limit, 
                framework, 
                target_ids
            )

    @staticmethod
    def _fetch_data_batch_tx(tx, node_label, condition, skip, limit, framework, target_ids=None):
        neighbor_condition = condition.replace("n.", "m.")
        
        query = f"MATCH (n:{node_label}) {condition} WITH n ORDER BY n.patientId SKIP $skip LIMIT $limit OPTIONAL MATCH (n)<-[e]-(m:{node_label}) {neighbor_condition} RETURN id(n) AS seed_id, n.label AS seed_label, n.features AS seed_f1, id(m) AS neighbor_id, m.features AS neighbor_f1, e.weight AS edge_weight"
        
        params = {"skip": skip, "limit": limit}
        if target_ids is not None:
            params["ids"] = target_ids
            
        result = tx.run(query, **params)
        
        node_features = []
        node_labels = []
        node_mask = []
        id_to_index = {}
        edge_sources = []
        edge_targets = []
        edge_weights = []
        
        if result.peek() is None or len(result.peek()) == 0:
            return np.array([]), np.array([])
        
        for record in result:
            seed_id = record["seed_id"]
            
            if seed_id not in id_to_index:
                id_to_index[seed_id] = len(node_features)
                f1 = record["seed_f1"] or []
                node_features.append(f1)
                
                label = record["seed_label"]
                node_labels.append(label if label is not None else np.nan)
                node_mask.append(True)
                
            neighbor_id = record["neighbor_id"]
            
            if neighbor_id is not None:
                if neighbor_id not in id_to_index:
                    id_to_index[neighbor_id] = len(node_features)
                    f1_n = record["neighbor_f1"] or []
                    node_features.append(f1_n)
                    
                    node_labels.append(np.nan)
                    node_mask.append(False)
                    
                edge_sources.append(id_to_index[neighbor_id])
                edge_targets.append(id_to_index[seed_id])
                
                weight = record["edge_weight"]
                edge_weights.append(weight if weight is not None else 1.0)
                
        features_arr = np.array(node_features, dtype=np.float32)
        labels_arr = np.array(node_labels, dtype=np.float32)
        edge_index_arr = np.array([edge_sources, edge_targets], dtype=np.int64)
        edge_weights_arr = np.array(edge_weights, dtype=np.float32)
        mask_arr = np.array(node_mask, dtype=bool)

        final_features = framework.get_features(features_arr, edge_index_arr, edge_weights_arr, mask=mask_arr, is_training=False)
        final_features = np.concatenate([f.cpu() for f in final_features], axis=1)
        
        extracted_labels = labels_arr[mask_arr]
        valid_mask = ~np.isnan(extracted_labels)
        
        return final_features[valid_mask], extracted_labels[valid_mask]

    def has_sbc_nodes(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n:SBC_TRAIN) RETURN COUNT(n) AS count").single()
            return result["count"] > 0      

    def has_mimic_nodes(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n:MIMIC_TRAIN) RETURN COUNT(n) AS count").single()
            return result["count"] > 0