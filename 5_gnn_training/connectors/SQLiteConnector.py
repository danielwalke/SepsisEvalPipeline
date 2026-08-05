import configparser
import os
import sqlite3
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm

class SQLiteConnector:
    def __init__(self, db_path=None):
        # timeout=60 ensures workers don't immediately lock out if database is busy
        config = configparser.ConfigParser()
        config_paths = ['/app/config/config.ini', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config.ini')), 'config.ini']
        config.read(config_paths)
        panel_name = config['PANEL']["panel_name"]
        if db_path is None:
            if os.path.exists('/app') and os.access('/app', os.W_OK):
                db_path = f"/app/db/{panel_name}/mimic_sbc_graph.db"
            else:
                db_path = os.path.abspath(f"4_db_upload/sqlite/db/{panel_name}/mimic_sbc_graph.db")
        self.conn = sqlite3.connect(db_path, timeout=60.0)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA synchronous = NORMAL;")
        
    def close(self):
        self.conn.close()
        
    def get_patient_ids(self, split_name):
        self.cursor.execute(f"SELECT DISTINCT patientId FROM {split_name}_nodes")
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_sbc_patient_train_ids(self): return self.get_patient_ids('SBC_TRAIN')
    def get_sbc_patient_test_ids(self): return self.get_patient_ids('SBC_TEST')
    def get_sbc_patient_ext_test_ids(self): return self.get_patient_ids('SBC_EXT_TEST')

    def get_mimic_patient_train_ids(self): return self.get_patient_ids('MIMIC_TRAIN')
    def get_mimic_patient_test_ids(self): return self.get_patient_ids('MIMIC_TEST')
    def get_mimic_patient_val_ids(self): return self.get_patient_ids('MIMIC_VAL')

    def get_ids_from_patient_ids(self, patient_ids, split_name):
        if not patient_ids:
            return []
        # Chunking to avoid SQLite variable limits (999 or 32766 depending on version)
        ids = []
        chunk_size = 900 
        for i in range(0, len(patient_ids), chunk_size):
            chunk = patient_ids[i:i + chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            self.cursor.execute(
                f"SELECT id FROM {split_name}_nodes WHERE patientId IN ({placeholders})", 
                chunk
            )
            ids.extend([row[0] for row in self.cursor.fetchall()])
        return ids

    def get_sbc_train_ids(self, patient_ids): return self.get_ids_from_patient_ids(patient_ids, 'SBC_TRAIN')
    def get_sbc_test_ids(self, patient_ids): return self.get_ids_from_patient_ids(patient_ids, 'SBC_TEST')
    def get_sbc_ext_test_ids(self, patient_ids): return self.get_ids_from_patient_ids(patient_ids, 'SBC_EXT_TEST')

    def get_mimic_train_ids(self, patient_ids): return self.get_ids_from_patient_ids(patient_ids, 'MIMIC_TRAIN')
    def get_mimic_test_ids(self, patient_ids): return self.get_ids_from_patient_ids(patient_ids, 'MIMIC_TEST')
    def get_mimic_val_ids(self, patient_ids): return self.get_ids_from_patient_ids(patient_ids, 'MIMIC_VAL')

    def get_seeded_subgraphs(self, batch_seeds, hops_limits, split_name):
        limit_1, limit_2 = hops_limits
        nodes_table = f"{split_name}_nodes"
        edges_table = f"{split_name}_edges"

        all_nodes = set(batch_seeds)
        current_targets = list(batch_seeds)

        # Hop 1 (Sample neighbors pointing TO our current targets)
        if limit_1 > 0 and current_targets:
            placeholders = ','.join(['?'] * len(current_targets))
            query1 = f"""
                SELECT source FROM (
                    SELECT source, ROW_NUMBER() OVER(PARTITION BY target ORDER BY source) as rn
                    FROM {edges_table} WHERE target IN ({placeholders})
                ) WHERE rn <= ?
            """
            self.cursor.execute(query1, current_targets + [limit_1])
            hop1_nodes = [row[0] for row in self.cursor.fetchall()]
            all_nodes.update(hop1_nodes)
            current_targets = list(set(hop1_nodes)) 

        # Hop 2
        if limit_2 > 0 and current_targets:
            placeholders = ','.join(['?'] * len(current_targets))
            query2 = f"""
                SELECT source FROM (
                    SELECT source, ROW_NUMBER() OVER(PARTITION BY target ORDER BY source) as rn
                    FROM {edges_table} WHERE target IN ({placeholders})
                ) WHERE rn <= ?
            """
            self.cursor.execute(query2, current_targets + [limit_2])
            hop2_nodes = [row[0] for row in self.cursor.fetchall()]
            all_nodes.update(hop2_nodes)

        # Use a TEMP table to reliably extract the induced subgraph without hitting variable limits
        self.cursor.execute("CREATE TEMP TABLE IF NOT EXISTS current_batch (id INTEGER PRIMARY KEY)")
        self.cursor.execute("DELETE FROM current_batch")
        self.cursor.executemany("INSERT INTO current_batch (id) VALUES (?)", [(n,) for n in all_nodes])

        # Get Induced Edges
        self.cursor.execute(f"""
            SELECT e.source, e.target, coalesce(e.weight, 1.0) 
            FROM {edges_table} e
            JOIN current_batch s ON e.source = s.id
            JOIN current_batch t ON e.target = t.id
        """)
        edge_data = self.cursor.fetchall()

        # Get Nodes
        # Fallback to 'features' if 'features_scaled' is missing/null, to prevent runtime errors
        self.cursor.execute(f"""
            SELECT n.id, coalesce(n.features_scaled, n.features), n.label 
            FROM {nodes_table} n
            JOIN current_batch b ON n.id = b.id
            WHERE n.id > -1
        """)
        node_data = self.cursor.fetchall()

        if not node_data or not edge_data:
            return Data()

        # Parse Data - Unpack BLOBs to float32 NumPy arrays
        node_ids = [row[0] for row in node_data]
        features = [np.frombuffer(row[1], dtype=np.float32) for row in node_data]
        labels = [int(row[2]) for row in node_data]
        
        edge_np = np.array(edge_data, dtype=np.float64)

        # PyTorch Geometric Construction
        global_src = torch.from_numpy(edge_np[:, 0].astype(np.int64))
        global_dst = torch.from_numpy(edge_np[:, 1].astype(np.int64))
        weights    = torch.from_numpy(edge_np[:, 2].astype(np.float32)).unsqueeze(1)

        all_node_ids = torch.tensor(node_ids, dtype=torch.long)
        
        # Stack the list of arrays into a single NumPy array, then convert to PyTorch
        x = torch.from_numpy(np.stack(features))
        y = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

        sorted_ids, sort_idx = torch.sort(all_node_ids)
        x = x[sort_idx]
        y = y[sort_idx]

        src_local = torch.searchsorted(sorted_ids, global_src)
        dst_local = torch.searchsorted(sorted_ids, global_dst)
        edge_index = torch.stack([src_local, dst_local], dim=0)
        
        batch_seeds_tensor = torch.tensor(batch_seeds, dtype=torch.long)
        batch_mask = torch.isin(sorted_ids, batch_seeds_tensor)        

        return Data(x=x, edge_index=edge_index, edge_attr=weights, y=y, batch_mask=batch_mask)
    
    def get_full_batch_graph(self, split_name):
        """
        Retrieves the entire graph for a given split without neighbor sampling.
        Useful for full-batch GNN training/evaluation.
        """
        nodes_table = f"{split_name}_nodes"
        edges_table = f"{split_name}_edges"

        # 1. Fetch All Edges for the split
        self.cursor.execute(f"""
            SELECT source, target, coalesce(weight, 1.0) 
            FROM {edges_table}
        """)
        edge_data = self.cursor.fetchall()

        # 2. Fetch All Nodes for the split
        self.cursor.execute(f"""
            SELECT id, coalesce(features_scaled, features), label 
            FROM {nodes_table}
            WHERE id > -1
        """)
        node_data = self.cursor.fetchall()

        # If there are no nodes, return an empty Data object
        if not node_data:
            return Data()

        # 3. Parse Node Data
        node_ids = [row[0] for row in node_data]
        features = [np.frombuffer(row[1], dtype=np.float32) for row in node_data]
        labels = [int(row[2]) for row in node_data]

        # Convert to PyTorch tensors
        all_node_ids = torch.tensor(node_ids, dtype=torch.long)
        x = torch.from_numpy(np.stack(features))
        y = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

        # Sort nodes so we can use searchsorted for O(log N) global-to-local ID mapping
        sorted_ids, sort_idx = torch.sort(all_node_ids)
        x = x[sort_idx]
        y = y[sort_idx]

        # 4. Parse Edge Data and Create Edge Index
        if edge_data:
            edge_np = np.array(edge_data, dtype=np.float64)
            global_src = torch.from_numpy(edge_np[:, 0].astype(np.int64))
            global_dst = torch.from_numpy(edge_np[:, 1].astype(np.int64))
            weights    = torch.from_numpy(edge_np[:, 2].astype(np.float32)).unsqueeze(1)

            # Map the global DB IDs to the local index (0 to N-1)
            src_local = torch.searchsorted(sorted_ids, global_src)
            dst_local = torch.searchsorted(sorted_ids, global_dst)
            edge_index = torch.stack([src_local, dst_local], dim=0)
        else:
            # Handle the edge case where a graph has nodes but completely isolated components (no edges)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            weights = torch.empty((0, 1), dtype=torch.float32)

        # In a full batch scenario, all nodes belong to the batch
        batch_mask = torch.ones(x.size(0), dtype=torch.bool)

        return Data(x=x, edge_index=edge_index, edge_attr=weights, y=y, batch_mask=batch_mask)

    def scale_and_add_positional_encodings(self, train_split_name, *test_split_names, chunk_size=10000):
        """
        Scales features and adds positional encodings using memory-efficient chunking 
        and high-speed NumPy vectorization.
        """
        
        print(f"Calculating global mins and maxs from {train_split_name} in chunks...")
        
        # 1. Get total row count for progress tracking
        self.cursor.execute(f"SELECT COUNT(*) FROM {train_split_name}_nodes WHERE id > -1")
        total_train_nodes = self.cursor.fetchone()[0]
        
        # 2. Streaming Min/Max calculation
        self.cursor.execute(f"SELECT features FROM {train_split_name}_nodes WHERE id > -1")
        mins, maxs = None, None
        
        with tqdm(total=total_train_nodes, desc="Computing Min/Max") as pbar:
            while True:
                rows = self.cursor.fetchmany(chunk_size)
                if not rows: break
                
                # Stack chunk and compute min/max for this batch
                chunk_features = np.stack([np.frombuffer(row[0], dtype=np.float32) for row in rows])
                chunk_mins = chunk_features.min(axis=0)
                chunk_maxs = chunk_features.max(axis=0)
                
                if mins is None:
                    mins, maxs = chunk_mins, chunk_maxs
                else:
                    mins = np.minimum(mins, chunk_mins)
                    maxs = np.maximum(maxs, chunk_maxs)
                
                pbar.update(len(rows))
                
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0  # Prevent division by zero
        
        print("Feature mins:", mins)
        print("Feature maxs:", maxs)

        splits_to_scale = [train_split_name] + list(test_split_names)
        
        # 3. Create a separate cursor for updates to avoid locking the SELECT stream
        update_cursor = self.conn.cursor()
        
        for split_name in splits_to_scale:
            print(f"\nScaling features for {split_name}...")
            
            # Ensure target column exists
            try:
                update_cursor.execute(f"ALTER TABLE {split_name}_nodes ADD COLUMN features_scaled BLOB")
            except sqlite3.OperationalError:
                pass 
            
            # Get count of unscaled nodes
            self.cursor.execute(f"SELECT COUNT(*) FROM {split_name}_nodes WHERE id > -1 AND features_scaled IS NULL")
            total_split_nodes = self.cursor.fetchone()[0]
            
            if total_split_nodes == 0:
                print(f"No rows need updating for {split_name}.")
                continue
                
            self.cursor.execute(f"SELECT id, features, pos_encodings FROM {split_name}_nodes WHERE id > -1 AND features_scaled IS NULL") #, pos_encodings
            
            with tqdm(total=total_split_nodes, desc=f"Scaling {split_name}") as pbar:
                while True:
                    rows = self.cursor.fetchmany(chunk_size)
                    if not rows: break
                    
                    # Unpack Python data into isolated lists
                    node_ids = [row[0] for row in rows]
                    
                    # Build 2D NumPy Matrices for the whole chunk
                    chunk_feats = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
                    
                    # Handle positional encodings (fallback to zeros if missing)
                    feat_dim = chunk_feats.shape[1]
                    chunk_pos_enc = np.stack([
                        np.frombuffer(row[2], dtype=np.float32) if row[2] else np.zeros(feat_dim, dtype=np.float32) 
                        for row in rows
                    ])
                    
                    # VECTORIZED MATH: Calculate all rows instantly in C
                    chunk_scaled = ((chunk_feats - mins) / ranges) + chunk_pos_enc
                    
                    # Repack the results back to bytes for SQLite
                    update_data = [
                        (feat_row.tobytes(), n_id) 
                        for feat_row, n_id in zip(chunk_scaled, node_ids)
                    ]
                    
                    # Execute and commit chunk
                    update_cursor.executemany(f"UPDATE {split_name}_nodes SET features_scaled = ? WHERE id = ?", update_data)
                    self.conn.commit()
                    
                    pbar.update(len(rows))
            
        print("\nScaled features and added positional encodings for all specified splits.")

    def has_nodes(self, split_name):
        self.cursor.execute(f"SELECT COUNT(1) FROM {split_name}_nodes")
        return self.cursor.fetchone()[0] > 0
        
    def has_sbc_nodes(self): return self.has_nodes('SBC_TRAIN')
    def has_mimic_nodes(self): return self.has_nodes('MIMIC_TRAIN')

    def get_node_count(self, split_name):
        self.cursor.execute(f"SELECT COUNT(1) FROM {split_name}_nodes")
        return self.cursor.fetchone()[0]
    
    def get_pos_weight(self, split_name):
        self.cursor.execute(f"SELECT SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END), SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) FROM {split_name}_nodes")
        pos_count, neg_count = self.cursor.fetchone()
        if pos_count == 0:
            return float('inf')  
        return neg_count / pos_count