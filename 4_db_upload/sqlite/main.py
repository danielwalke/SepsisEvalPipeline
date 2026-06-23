import sqlite3
import csv
import os
import array
from SQLiteQueries import SQLiteQueries

def clear_database(cursor):
    """Drops all tables in the database to simulate a clean slate."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        cursor.execute(f"DROP TABLE {table_name[0]}")
    print("Cleared existing data in the database.")

def process_and_upload_split(conn, csv_dir, cursor, split_label, file_prefix):
    """Helper function to process CSVs and upload a specific split."""
    queries = SQLiteQueries(split_label)
    
    # 1. Initialize Schema
    cursor.executescript(queries.get_tables_creation_query())
    cursor.executescript(queries.get_edge_index_query())
    
    # 2. Upload Nodes
    node_data = []
    with open(f"{os.path.join(csv_dir, file_prefix)}_nodes.csv", 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            label = 1 if row['y'] == 'Sepsis' else 0
            
            # Clean and parse features list, then serialize to bytes
            features_raw = row['X'].replace('[', '').replace(']', '').split(',')
            features_list = [float(x.strip()) for x in features_raw if x.strip()]
            # 'f' denotes a C-style float (32-bit). Use 'd' if you need 64-bit doubles.
            features_blob = array.array('f', features_list).tobytes()
            
            node_data.append((
                i,                      # id (corresponds to linenumber offset)
                label,                  # label
                features_blob,          # features (as bytes)
                int(row['patientId']),  # patientId
                float(row['time']),     # time
                int(row['hadmId'])      # hadmId
            ))
            
    cursor.executemany(queries.get_nodes_insertion_query(), node_data)
    conn.commit()
    print(f"Nodes for {split_label} uploaded.")

    # 3. Upload Edges
    edge_data = []
    with open(f"{os.path.join(csv_dir, file_prefix)}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            edge_data.append((
                int(row[0]),    # source
                int(row[1]),    # target
                float(row[2])   # weight
            ))
            
    cursor.executemany(queries.get_edges_insertion_query(), edge_data)
    conn.commit()
    print(f"Edges for {split_label} uploaded.")

    # 4. Upload Positional Encodings
    pos_data = []
    with open(f"{os.path.join(csv_dir, file_prefix)}_pos_encodings.csv", 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Clean and parse positional encodings, then serialize to bytes
            pos_raw = row['pos_encodings'].replace('[', '').replace(']', '').split(',')
            pos_list = [float(x.strip()) for x in pos_raw if x.strip()]
            pos_blob = array.array('f', pos_list).tobytes()
            
            # Note the order: (pos_encodings, id) to match the UPDATE query
            pos_data.append((pos_blob, i))
            
    cursor.executemany(queries.get_pos_enc_update_query(), pos_data)
    conn.commit()
    print(f"Positional encodings for {split_label} uploaded.")


if __name__ == "__main__":
    csv_dir = os.environ.get("CSV_DIR", ".")
    db_path = os.environ.get("DB_PATH", "mimic_sbc_graph.db")
    
    print(f"Connecting to SQLite database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clean up old data
    clear_database(cursor)
    conn.commit()
    
    # Process MIMIC Splits
    mimic_split_dict = {
        "train": "MIMIC_TRAIN",
        "test": "MIMIC_TEST",
        "val": "MIMIC_VAL"
    }
    
    for split, split_label in mimic_split_dict.items():
        print(f"\nAttempting to upload {split} data to SQLite database...")
        try:
            file_prefix = f"mimic_{split}"
            process_and_upload_split(conn, csv_dir, cursor, split_label, file_prefix)
        except Exception as e:
            print(f"Skipping {split} split. A file was missing or invalid. Error details: {e}")
            conn.rollback() # Roll back transaction on error
            continue

    # Process SBC Splits
    split_dict = {
            "": "SBC_TRAIN",
            "_validation": "SBC_TEST",
            "_ext_validation": "SBC_EXT_TEST"
    }
    for split, split_label in split_dict.items():
        print(f"\nAttempting to upload SBC{split} data to SQLite database...")
        try:
            file_prefix = f"sbc{split}"
            process_and_upload_split(conn, csv_dir, cursor, split_label, file_prefix)
        except Exception as e:
            print(f"Skipping SBC{split} split. A file was missing or invalid. Error details: {e}")
            conn.rollback() # Roll back transaction on error
            continue