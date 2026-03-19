class Neo4jQueries:
        def __init__(self, split):
            self.split = split


        def get_constraint_query(self):
                CREATE_CONSTRAINT_QUERY = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{self.split}) REQUIRE n.id IS UNIQUE"
                return CREATE_CONSTRAINT_QUERY
        
        def get_nodes_creation_query(self):
            return f"""
                LOAD CSV WITH HEADERS FROM $file AS line
                WITH line, linenumber() AS index
                CALL(line, index) {{
                        WITH line, index
                        CREATE (p:{self.split} {{id: index - 2}})
                        SET p.label = CASE WHEN line.y = 'Sepsis' THEN 1 ELSE 0 END,
                        p.features = [x IN split(trim(replace(replace(line.X, '[', ''), ']', '')), ',') | toFloat(trim(x))],
                        p.patientId = toInteger(line.patientId),
                        p.time = toFloat(line.time)
                }} IN TRANSACTIONS OF 500 ROWS
                """ ## -2 because linenumber starts at 1 and we have a header row anid the edge index starts at 0
                
        def get_node_id_index_query(self):
                return f"CREATE INDEX IF NOT EXISTS FOR (n:{self.split}) ON (n.id)"
        
        def get_edges_creation_query(self):
            return f"""
                LOAD CSV FROM $file AS line
                WITH line, linenumber() AS index
                WHERE index <> 1
                CALL(line, index) {{
              
                   WITH line
                    MATCH (s:{self.split} {{id: toInteger(line[0])}}), (t:{self.split} {{id: toInteger(line[1])}})
                    CREATE (s)-[r:connects {{weight: toFloat(line[2])}}]->(t)
                }} IN TRANSACTIONS OF 100000 ROWS
                """
        def get_pos_enc_creation_query(self):
            return f"""
            LOAD CSV WITH HEADERS FROM $file AS line
            WITH line, linenumber() AS index
            CALL (line, index) {{
                WITH line, index,
                     split(trim(replace(replace(line.pos_encodings, '[', ''), ']', '')), ',') AS pos_encodings
                MATCH (p:{self.split} {{id: index - 2}})
                SET p.pos_encodings = [x IN pos_encodings | toFloat(trim(x))]
            }} IN TRANSACTIONS OF 500 ROWS
            """