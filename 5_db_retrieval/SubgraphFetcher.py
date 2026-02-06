import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from neo4j import GraphDatabase
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from TestGNN_NeighborSamplingNeo4j import ImprovedTwoLayerGNN

# --- 1. Memory Calculator (Same as before) ---
def calculate_loader_params(max_ram_gb, batch_size, num_workers, num_hops, neighbor_limit, num_features=7):
    if neighbor_limit == 1:
        max_nodes_per_seed = num_hops + 1
    else:
        max_nodes_per_seed = (neighbor_limit**(num_hops + 1) - 1) / (neighbor_limit - 1)
    
    max_edges_per_seed = max_nodes_per_seed * 3
    node_bytes = (num_features * 4) + 4 + 8 
    edge_bytes = 8 + 8 + 4
    subgraph_bytes = (max_nodes_per_seed * node_bytes) + (max_edges_per_seed * edge_bytes)
    total_bytes_per_seed = subgraph_bytes * 4
    total_bytes_per_batch = total_bytes_per_seed * batch_size
    max_ram_bytes = max_ram_gb * (1024**3)
    max_batches_in_ram = max_ram_bytes / total_bytes_per_batch
    safe_batches = int(max_batches_in_ram * 0.8)
    
    if num_workers > 0:
        prefetch_factor = safe_batches // num_workers
    else:
        prefetch_factor = safe_batches
    return max(2, min(prefetch_factor, 50))

# --- 2. Cypher Query (Same as before) ---
def get_cypher_query(split='train'):
    split = split.upper()
    # Added coalesce for weights to default to 1.0 if missing
    # Uncommented rand() for necessary stochasticity
    return f"""
UNWIND $seed_ids AS seedId
MATCH (seed:{split})
WHERE seed.id = seedId

CALL(seed){{
    MATCH (seed)<--(n1)
    WITH n1
    ORDER BY n1.id
    LIMIT $limit_1
    RETURN collect(n1) AS hop1_nodes
}}

CALL(hop1_nodes) {{
    UNWIND hop1_nodes AS h1
    MATCH (h1)<--(h2)
    WITH h2
    ORDER BY h2.id
    LIMIT $limit_2
    RETURN collect(DISTINCT h2) AS hop2_nodes
}}

WITH seed, hop1_nodes, hop2_nodes, (hop1_nodes + hop2_nodes + [seed]) AS all_sampled_nodes
UNWIND all_sampled_nodes AS n
WITH DISTINCT n AS sampledNode, all_sampled_nodes

MATCH (sampledNode)-[r]->(target)
WHERE target IN all_sampled_nodes

WITH collect(DISTINCT sampledNode) AS uniqueNodes, collect(DISTINCT r) AS uniqueEdges
RETURN 
    [n IN uniqueNodes | n.id] AS node_ids,
    [n IN uniqueNodes | n.features_scaled] AS features,
    [n IN uniqueNodes | toInteger(n.label)] AS labels,
    [e IN uniqueEdges | [startNode(e).id, endNode(e).id, coalesce(e.weight, 1.0)]] AS edge_data
    """
    ## TODO check consistent id usage or edge direction?!

import torch

def assert_edges_in_superset(superset_index, subset_index):
    max_val = max(superset_index.max(), subset_index.max()) + 1
    
    superset_1d = superset_index[0] * max_val + superset_index[1]
    subset_1d = subset_index[0] * max_val + subset_index[1]
    
    mask = torch.isin(subset_1d, superset_1d)
    all_present = mask.all()
    
    assert all_present, "Edge assertion failed: Not all edges are in the target index"
# --- 3. The Corrected Dataset ---
class Neo4jGraphDataset(Dataset):
    def __init__(self, all_training_seed_ids, hops_limits, batch_size, split='train'):
        self.seed_ids = all_training_seed_ids
        self.hops_limits = hops_limits
        self.batch_size = batch_size
        self.driver = None 
        self.split = split
        self.query = get_cypher_query(split=self.split)

    def _init_worker_driver(self):
        if self.driver is None:
            # Tip: Use 'bolt' or 'neo4j' scheme depending on your cluster setup
            self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
            
    def __len__(self):
        return (len(self.seed_ids) + self.batch_size - 1) // self.batch_size

    def shuffle(self):
        """
        Crucial: Call this manually before the DataLoader iterator is created 
        at the start of each epoch loop.
        """
        random.shuffle(self.seed_ids)

    def __getitem__(self, idx):
        self._init_worker_driver()
        
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.seed_ids))
        batch_seeds = self.seed_ids[start:end]
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    self.query, 
                    seed_ids=batch_seeds, 
                    limit_1=self.hops_limits[0], 
                    limit_2=self.hops_limits[1]
                )
                record = result.single()
        except Exception as e:
            print(f"Error fetching batch {idx}: {e}")
            return Data() # Return empty Data instead of None

        # Logic check: 'record' might be None if query returns no rows
        if not record or not record['node_ids']: 
            return Data() 

        edge_np = np.array(record['edge_data'], dtype=np.float64)
        
        # Handle single isolated nodes (no edges returned)
        if edge_np.size == 0:
            # Even if no edges, we might have nodes. 
            # If your model needs edges, return empty.
            return Data()

        global_src = torch.from_numpy(edge_np[:, 0].astype(np.int64))
        global_dst = torch.from_numpy(edge_np[:, 1].astype(np.int64))
        weights    = torch.from_numpy(edge_np[:, 2].astype(np.float32)).unsqueeze(1)

        all_node_ids = torch.tensor(record['node_ids'], dtype=torch.long)
        x = torch.tensor(record['features'], dtype=torch.float)
        y = torch.tensor(record['labels'], dtype=torch.float).unsqueeze(1)

        # Optimization: Sort once, use for both x/y reordering and searchsorted
        sorted_ids, sort_idx = torch.sort(all_node_ids)
        x = x[sort_idx]
        y = y[sort_idx]

        src_local = torch.searchsorted(sorted_ids, global_src)
        dst_local = torch.searchsorted(sorted_ids, global_dst)
        edge_index = torch.stack([src_local, dst_local], dim=0)

        # Create Train Mask
        batch_seeds_tensor = torch.tensor(batch_seeds, dtype=torch.long)
        # isin requires both tensors to be on same device (CPU here), which they are.
        train_mask = torch.isin(sorted_ids, batch_seeds_tensor)
       
        # --- DEBUGGING EDGE CONSISTENCY for global_src and global_dst ---
        # Uncomment below to verify sampled edges contain all ground truth edges among training nodes 
        # from torch_geometric.utils import sort_edge_index
        # import pandas as pd
        
        # gt_edges = pd.read_csv("./3_graph_construction/data/sbc_edges.csv")
        # gt_edge_index = torch.tensor(gt_edges[["source", "target"]].values.T, dtype=torch.long)
        # sorted_gt_edge_index = sort_edge_index(gt_edge_index)
        
        # sampled_edge_index = torch.stack([global_src, global_dst], dim=0).long()
        # sorted_sampled_edge_index = sort_edge_index(sampled_edge_index)
        
        
        # assert_edges_in_superset(sorted_gt_edge_index, sorted_sampled_edge_index)
        

        return Data(x=x, edge_index=edge_index, edge_attr=weights, y=y, train_mask=train_mask)

    
def eval_model(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            # Check for empty batch (failed fetch or isolated nodes)
            if batch.x is None: 
                continue
                
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Use mask to only evaluate on the specific Seed nodes of this batch
            if batch.train_mask.sum() == 0:
                continue

            target = batch.y[batch.train_mask]
            logits = out[batch.train_mask]
            probs = torch.sigmoid(logits)
            
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    try:
        # Check if we actually collected data to avoid ROC error
        if not all_labels: return 0.0
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0
    return auroc

def scale_and_add_pos_enc_to_features(driver, train_split_name, val_split_name, ext_val_split_name=None):
    with driver.session() as session:
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
            WHERE n.features_scaled IS NULL
            CALL {
                WITH n
                SET n.features_scaled = [i IN range(0, size(n.features)-1) | 
                    CASE 
                        WHEN ($maxs[i] - $mins[i]) = 0 THEN 0.0 + n.pos_encodings[i]
                        ELSE (n.features[i] - $mins[i]) / ($maxs[i] - $mins[i]) + n.pos_encodings[i]
                    END
                ]
            } IN TRANSACTIONS OF 2000 ROWS
        """

        session.run(f"MATCH (n:{train_split_name}) {scale_query}", mins=mins, maxs=maxs)

        if val_split_name:
            session.run(f"MATCH (n:{val_split_name}) {scale_query}", mins=mins, maxs=maxs)
        if ext_val_split_name:
            session.run(f"MATCH (n:{ext_val_split_name}) {scale_query}", mins=mins, maxs=maxs)
        print("Scaled features and added positional encodings.")

## TODO theoretical fix: Add pos. enodings
if __name__ == '__main__':
    BATCH_SIZE = 512*10
    NUM_WORKERS = 6
    HOPS = 2
    LIMIT = 10
    MAX_RAM_GB = 20
    MAX_RAM_GB_TRAIN = MAX_RAM_GB * .8  # 80% for training
    MAX_RAM_GB_VAL = MAX_RAM_GB * .2    # 20% for validation
    
    # 1. Calculate Prefetch
    prefetch_train = 30# calculate_loader_params(MAX_RAM_GB_TRAIN, BATCH_SIZE, NUM_WORKERS, HOPS, LIMIT)
    prefetch_val = 30 #calculate_loader_params(MAX_RAM_GB_VAL, BATCH_SIZE, NUM_WORKERS, HOPS, LIMIT)
    
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    train_split_name = 'SBC_TRAIN'
    val_split_name = 'SBC_TEST'
    
    # scale_and_add_pos_enc_to_features(driver, train_split_name, val_split_name, ext_val_split_name="SBC_EXT_TEST")
    
    
    with driver.session() as session:
        train_seed_ids = session.run(f"MATCH (n:{train_split_name}) RETURN COLLECT(n.id) as train_ids").single().get("train_ids")
        val_seed_ids = session.run(f"MATCH (n:{val_split_name}) RETURN COLLECT(n.id) as val_ids").single().get("val_ids")
    train_dataset = Neo4jGraphDataset(train_seed_ids, hops_limits=[LIMIT, LIMIT], batch_size=BATCH_SIZE, split=train_split_name)
    val_dataset = Neo4jGraphDataset(val_seed_ids, hops_limits=[LIMIT, LIMIT], batch_size=BATCH_SIZE, split=val_split_name)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=None,  
        shuffle=True,     
        num_workers=NUM_WORKERS,
        prefetch_factor=prefetch_train, 
        persistent_workers=True
    )  
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        prefetch_factor=prefetch_val,
        persistent_workers=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = ImprovedTwoLayerGNN(in_channels=7, hidden_channels= 128, out_channels=1).to(device)
    pos_weight = torch.tensor([664.0]).to(device) #525.0 for mimic, 664.0 for sbc
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting Training with Prefetch Factor: {prefetch_train}")

    for epoch in range(100):
        
        model.train()
        total_loss = 0
        
        # --- STORAGE FOR AUROC ---
        all_labels = []
        all_probs = []

        for batch in train_loader:
            if batch is None: continue
            
            batch = batch.to(device)
            
            
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            
            target = batch.y[batch.train_mask]
            logits = out[batch.train_mask]
            
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")
        if epoch % 1 == 0:
            val_auroc = eval_model(model, val_loader, device)
            # train_auroc = eval_model(model, train_loader, device)
            # print(f"--- TRAIN AUROC: {train_auroc:.4f} --- VAL AUROC: {val_auroc:.4f} ---")
            print(f"--- VAL AUROC: {val_auroc:.4f} ---")