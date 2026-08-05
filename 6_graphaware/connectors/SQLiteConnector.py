import configparser
import os
import sqlite3
import re
import numpy as np

class SQLiteConnector:
    def __init__(self, db_path=None):
        """
        Initializes the SQLite connector.
        Reads the database path from the environment variable if not provided.
        """
        config = configparser.ConfigParser()
        config_paths = ['/app/config/config.ini', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config.ini')), 'config.ini']
        config.read(config_paths)
        panel_name = config['PANEL']["panel_name"] if 'PANEL' in config else "CBC"
        if db_path is None:
            if os.path.exists('/app') and os.access('/app', os.W_OK):
                db_path = f"/app/db/{panel_name}/mimic_sbc_graph.db"
            else:
                db_path = os.path.abspath(f"4_db_upload/sqlite/db/{panel_name}/mimic_sbc_graph.db")
        
        # timeout=60.0 ensures concurrent workers wait instead of immediately locking out
        self.conn = sqlite3.connect(db_path, timeout=60.0)
        self.cursor = self.conn.cursor()
        
        # Turn on Write-Ahead Logging for fast concurrent reads/writes
        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA synchronous = NORMAL;")
        
    def close(self):
        """Closes the underlying database connection safely."""
        self.conn.close()

    def _has_table(self, table_name):
        """Helper to verify if a table exists in the database."""
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
            (table_name,)
        )
        return self.cursor.fetchone() is not None

    def has_sbc_nodes(self):
        """Checks if the SBC_TRAIN_nodes table exists and contains records."""
        if not self._has_table("SBC_TRAIN_nodes"):
            return False
        self.cursor.execute("SELECT COUNT(1) FROM SBC_TRAIN_nodes")
        return self.cursor.fetchone()[0] > 0      

    def has_mimic_nodes(self):
        """Checks if the MIMIC_TRAIN_nodes table exists and contains records."""
        if not self._has_table("MIMIC_TRAIN_nodes"):
            return False
        self.cursor.execute("SELECT COUNT(1) FROM MIMIC_TRAIN_nodes")
        return self.cursor.fetchone()[0] > 0

    def get_patient_ids(self, split_name):
        """Returns distinct patient IDs for a given split label."""
        table_name = f"{split_name}_nodes"
        self.cursor.execute(f"SELECT DISTINCT patientId FROM {table_name}")
        return [row[0] for row in self.cursor.fetchall()]

    def get_ids(self, split_name):
        """Returns all primary internal IDs for a given split label."""
        table_name = f"{split_name}_nodes"
        self.cursor.execute(f"SELECT id FROM {table_name} WHERE id > -1")
        return [row[0] for row in self.cursor.fetchall()]

    def _get_ids_by_patient_filter(self, split_name, patient_ids):
        """Helper to fetch node IDs belonging to specific patient IDs using safe chunking."""
        if not patient_ids:
            return []
        
        table_name = f"{split_name}_nodes"
        ids = []
        chunk_size = 900  # Avoid SQLite variable limits per query
        
        for i in range(0, len(patient_ids), chunk_size):
            chunk = patient_ids[i:i + chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            self.cursor.execute(
                f"SELECT id FROM {table_name} WHERE patientId IN ({placeholders})", 
                chunk
            )
            ids.extend([row[0] for row in self.cursor.fetchall()])
        return ids

    def get_sbc_train_ids(self, train_patient_ids):
        """Backward compatible signature for fetch training subset IDs."""
        return self._get_ids_by_patient_filter("SBC_TRAIN", train_patient_ids)

    def get_sbc_val_ids(self, val_patient_ids):
        """Backward compatible signature for validation subset IDs."""
        return self._get_ids_by_patient_filter("SBC_TRAIN", val_patient_ids)

    def fetch_data_batch(self, node_label, condition, skip, limit, framework, target_ids=None):
        """
        Extracts feature batches using Temporary Tables to safely handle massive 
        Full-Batch scales (millions of nodes) without hitting SQLite limits.
        Utilizes NumPy vectorization for high-speed array reconstruction.
        """
        nodes_table = f"{node_label}_nodes"
        edges_table = f"{node_label}_edges"
        
        sql_condition = ""
        sql_params = []
        
        if condition and "WHERE" in condition.upper():
            if "$ids" in condition and target_ids:
                # 1. Temp Table for Target IDs (Bulletproof for Full-Batch scaling)
                self.cursor.execute("CREATE TEMP TABLE IF NOT EXISTS target_seeds_tmp (id INTEGER PRIMARY KEY)")
                self.cursor.execute("DELETE FROM target_seeds_tmp")
                self.cursor.executemany("INSERT INTO target_seeds_tmp (id) VALUES (?)", [(tid,) for tid in target_ids])
                
                # Replace placeholder with subquery targeting our temp table
                sql_condition = condition.replace("$ids", "(SELECT id FROM target_seeds_tmp)")
            else:
                sql_condition = condition
        
        # 2. Extract Seed/Target Nodes
        seed_query = f"""
            SELECT id, features, label, patientId
            FROM {nodes_table}
            {sql_condition}
            ORDER BY patientId
            LIMIT ? OFFSET ?
        """
        self.cursor.execute(seed_query, sql_params + [limit, skip])
        seed_rows = self.cursor.fetchall()
        
        if not seed_rows:
            return np.array([]), np.array([])
            
        seed_ids = [row[0] for row in seed_rows]
        
        # 3. Temp Table for Seed IDs to safely map induced edges
        self.cursor.execute("CREATE TEMP TABLE IF NOT EXISTS batch_seeds_tmp (id INTEGER PRIMARY KEY)")
        self.cursor.execute("DELETE FROM batch_seeds_tmp")
        self.cursor.executemany("INSERT INTO batch_seeds_tmp (id) VALUES (?)", [(sid,) for sid in seed_ids])
        
        # 4. Extract Edges using native SQL JOIN
        self.cursor.execute(f"""
            SELECT e.source, e.target, coalesce(e.weight, 1.0)
            FROM {edges_table} e
            JOIN batch_seeds_tmp t ON e.target = t.id
        """)
        edge_rows = self.cursor.fetchall()
        
        # 5. Extract Neighbors
        neighbor_ids = list(set(row[0] for row in edge_rows if row[0] not in seed_ids))
        neighbor_rows = []
        
        if neighbor_ids:
            # Chunking is safe here as neighbor sets retrieved in increments won't exceed limits
            for i in range(0, len(neighbor_ids), 900):
                chunk = neighbor_ids[i:i+900]
                placeholders = ','.join(['?'] * len(chunk))
                self.cursor.execute(f"""
                    SELECT id, features
                    FROM {nodes_table} WHERE id IN ({placeholders})
                """, chunk)
                neighbor_rows.extend(self.cursor.fetchall())

        # =====================================================================
        # NUMPY VECTORIZATION BLOCK (Replaces Python 'for' loops and dicts)
        # =====================================================================

        # Fast tuple unpacking using comprehensions
        seed_feats = [np.frombuffer(r[1], dtype=np.float32) if r[1] else [] for r in seed_rows]
        seed_labels = [r[2] if r[2] is not None else np.nan for r in seed_rows]

        neigh_feats = [np.frombuffer(r[1], dtype=np.float32) if r[1] else [] for r in neighbor_rows]
        neigh_labels = [np.nan] * len(neighbor_rows)



        # Concatenate nodes directly into memory
        features_arr = np.array(seed_feats + neigh_feats, dtype=np.float32)
        labels_arr = np.array(seed_labels + neigh_labels, dtype=np.float32)

        # Create boolean mask: True for seeds, False for neighbors
        mask_arr = np.zeros(len(features_arr), dtype=bool)
        mask_arr[:len(seed_rows)] = True  

        # Vectorized Edge Mapping
        if not edge_rows:
            edge_index_arr = np.empty((2, 0), dtype=np.int64)
            edge_weights_arr = np.array([], dtype=np.float32)
        else:
            edge_src = np.array([r[0] for r in edge_rows], dtype=np.int64)
            edge_dst = np.array([r[1] for r in edge_rows], dtype=np.int64)
            edge_weights_arr = np.array([r[2] for r in edge_rows], dtype=np.float32)

            # Create a unified ID lookup array
            all_ids = np.array(seed_ids + [r[0] for r in neighbor_rows], dtype=np.int64)

            # Map global database IDs to local array indices using fast binary search
            sort_idx = np.argsort(all_ids)
            sorted_ids = all_ids[sort_idx]

            src_local = sort_idx[np.searchsorted(sorted_ids, edge_src)]
            dst_local = sort_idx[np.searchsorted(sorted_ids, edge_dst)]

            # Stack into the final 2xN edge index matrix
            edge_index_arr = np.stack([src_local, dst_local])

        # =====================================================================
        # END VECTORIZATION BLOCK
        # =====================================================================

        # Process spatial feature aggregations via GraphAware Ensemble Framework
        final_features = framework.get_features(
            features_arr, 
            edge_index_arr, 
            edge_weights_arr, 
            mask=mask_arr, 
            is_training=False
        )
        
        # Handle tensor unwrapping to match NumPy expectations 
        final_features = np.concatenate([f.cpu().numpy() if hasattr(f, 'cpu') else f for f in final_features], axis=1)
        
        # Isolate seed evaluation targets
        extracted_labels = labels_arr[mask_arr]
        valid_mask = ~np.isnan(extracted_labels)
        
        return final_features[valid_mask], extracted_labels[valid_mask]
    
    def get_node_count(self):
        """Returns the total number of nodes in the database for progress estimation."""
        self.cursor.execute("SELECT COUNT(1) FROM (SELECT id FROM SBC_TRAIN_nodes UNION ALL SELECT id FROM MIMIC_TRAIN_nodes)")
        return self.cursor.fetchone()[0]