from torch.utils.data import Dataset
import random
from connectors.SQLiteConnector import SQLiteConnector
from connectors.Neo4jConnector import Neo4jConnector




class GraphDataset(Dataset):
    def __init__(self, connector, all_training_seed_ids, hops_limits, batch_size, split='train'):
        self.seed_ids = all_training_seed_ids
        self.hops_limits = hops_limits
        self.batch_size = batch_size
        self.split = split
        self.connector = connector

    def get_connector_class(self):
        db = "sqlite"
        if db == "sqlite":
            return SQLiteConnector
        elif db == "neo4j":
            return Neo4jConnector
        else:
            raise ValueError(f"Unsupported database type: {db}")

    def _init_worker_driver(self):
        # Re-initialize the connector for each worker to avoid SQLite connection issues
        if not hasattr(self, 'connector') or self.connector is None:
            self.connector = self.get_connector_class()()
            
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
        return self.connector.get_seeded_subgraphs(batch_seeds, self.hops_limits, split_name=self.split)
        
        