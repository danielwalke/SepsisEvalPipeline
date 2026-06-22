from neo4j import GraphDatabase
import os
import pandas as pd

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
with driver.session() as session:
    unique_hadm_ids = session.run("""MATCH (p)
WHERE p.hadmId IS NOT NULL
WITH DISTINCT p.hadmId AS hadmId
LIMIT 5
RETURN collect(hadmId) AS hadmIds""").single()["hadmIds"]  
    print(unique_hadm_ids)
