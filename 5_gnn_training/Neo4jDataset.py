from torch.utils.data import Dataset
from connectors.Neo4jConnector import Neo4jConnector
import random




class Neo4jGraphDataset(Dataset):
    def __init__(self, all_training_seed_ids, hops_limits, batch_size, split='train'):
        self.seed_ids = all_training_seed_ids
        self.hops_limits = hops_limits
        self.batch_size = batch_size
        self.split = split
        self.neo4j_connector = None

    def _init_worker_driver(self):
        if self.neo4j_connector is None:
            self.neo4j_connector = Neo4jConnector()
            
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
        return self.neo4j_connector.get_seeded_subgraphs(batch_seeds, self.hops_limits, split_name=self.split)
        
        