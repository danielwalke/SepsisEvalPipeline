from torch.utils.data import Dataset
import random
from connectors.SQLiteConnector import SQLiteConnector
from connectors.Neo4jConnector import Neo4jConnector

class GraphDataset(Dataset):
    def __init__(self, db_type, all_training_seed_ids, hops_limits, batch_size, split='train', use_full_batch=True):
        self.db_type = db_type
        self.seed_ids = all_training_seed_ids
        self.hops_limits = hops_limits
        self.batch_size = batch_size
        self.split = split
        self.use_full_batch = use_full_batch
        self.connector = None 

    def get_connector_class(self):
        if self.db_type == "sqlite":
            return SQLiteConnector
        elif self.db_type == "neo4j":
            return Neo4jConnector
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def _init_worker_driver(self):
        # This will now correctly fire on the first __getitem__ call in each worker process
        if self.connector is None:
            self.connector = self.get_connector_class()()
            
    def __len__(self):
        if self.use_full_batch:
            # If full batch, the dataset size is just 1.
            return 1
        else:
            # If mini-batching, return the number of chunks.
            return (len(self.seed_ids) + self.batch_size - 1) // self.batch_size

    def shuffle(self):
        """Call this manually before the DataLoader iterator is created."""
        if not self.use_full_batch:
            random.shuffle(self.seed_ids)

    def __getitem__(self, idx):
        self._init_worker_driver()        
        
        if self.use_full_batch:
            # idx will always be 0 because __len__ is 1
            return self.connector.get_full_batch_graph(split_name=self.split)
            
        else:
            start = idx * self.batch_size
            end = min(start + self.batch_size, len(self.seed_ids))
            batch_seeds = self.seed_ids[start:end]
            
            return self.connector.get_seeded_subgraphs(batch_seeds, self.hops_limits, split_name=self.split)