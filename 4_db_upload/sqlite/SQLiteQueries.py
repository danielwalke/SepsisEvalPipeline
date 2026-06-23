class SQLiteQueries:
    def __init__(self, split):
        self.split = split
        self.nodes_table = f"{self.split}_nodes"
        self.edges_table = f"{self.split}_edges"

    def get_tables_creation_query(self):
        """
        Creates the relational schema. 
        Note: The Node ID is automatically indexed by making it the PRIMARY KEY.
        Lists (features, pos_encodings) are serialized and stored as BLOBs.
        """
        return f"""
        BEGIN TRANSACTION;
        CREATE TABLE IF NOT EXISTS {self.nodes_table} (
            id INTEGER PRIMARY KEY,
            label INTEGER,
            features BLOB,
            patientId INTEGER,
            time REAL,
            hadmId INTEGER,
            pos_encodings BLOB,
            features_scaled BLOB
        );
        CREATE TABLE IF NOT EXISTS {self.edges_table} (
            source INTEGER,
            target INTEGER,
            weight REAL,
            FOREIGN KEY(source) REFERENCES {self.nodes_table}(id),
            FOREIGN KEY(target) REFERENCES {self.nodes_table}(id)
        );
        COMMIT;
        """
        
    def get_edge_index_query(self):
        """
        CRITICAL: Creates indexes on source and target columns in the edge table.
        This compensates for SQLite's lack of index-free adjacency.
        """
        return f"""
        BEGIN TRANSACTION;
        CREATE INDEX IF NOT EXISTS idx_{self.edges_table}_source ON {self.edges_table} (source);
        CREATE INDEX IF NOT EXISTS idx_{self.edges_table}_target ON {self.edges_table} (target);
        COMMIT;
        """
    
    def get_nodes_insertion_query(self):
        """
        Returns a parameterized query for executemany().
        Expected tuple format: (id, label, features_blob, patientId, time, hadmId)
        """
        return f"""
        INSERT INTO {self.nodes_table} (id, label, features, patientId, time, hadmId)
        VALUES (?, ?, ?, ?, ?, ?)
        """
            
    def get_edges_insertion_query(self):
        """
        Returns a parameterized query for executemany().
        Expected tuple format: (source_id, target_id, weight)
        """
        return f"""
        INSERT INTO {self.edges_table} (source, target, weight)
        VALUES (?, ?, ?)
        """

    def get_pos_enc_update_query(self):
        """
        Returns a parameterized query for executemany() to update nodes.
        Expected tuple format: (pos_encodings_blob, id)
        """
        return f"""
        UPDATE {self.nodes_table}
        SET pos_encodings = ?
        WHERE id = ?
        """

    def get_self_loops_creation_query(self):
        """
        Creates self-loop edges purely via SQL logic, similar to the Neo4j equivalent.
        """
        return f"""
        INSERT INTO {self.edges_table} (source, target, weight)
        SELECT id, id, 1.0 FROM {self.nodes_table};
        """