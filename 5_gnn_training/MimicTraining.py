from GraphDataset import GraphDataset
from connectors.SQLiteConnector import SQLiteConnector
from connectors.Neo4jConnector import Neo4jConnector
from Dataloader import Dataloader

class Mimic_Training:
    def __init__(self, hops, db_type, MAX_RAM_GB, BATCH_SIZE, NUM_WORKERS, LIMIT, use_full_batch=True):
        self.db_type = db_type
        self.connector = SQLiteConnector() if db_type == "sqlite" else Neo4jConnector()
        self.has_mimic_nodes = self.connector.has_mimic_nodes()
        self.MAX_RAM_GB = MAX_RAM_GB
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_WORKERS = NUM_WORKERS
        self.LIMIT = LIMIT
        self.hops = hops
        self.use_full_batch = use_full_batch

    def get_dataloaders(self):
        self.connector.scale_and_add_positional_encodings('MIMIC_TRAIN', 'MIMIC_VAL', 'MIMIC_TEST')

        train_seed_patient_ids = self.connector.get_mimic_patient_train_ids()
        val_seed_patient_ids = self.connector.get_mimic_patient_val_ids()
        test_seed_patient_ids = self.connector.get_mimic_patient_test_ids()
    
        train_ids = self.connector.get_mimic_train_ids(train_seed_patient_ids)
        val_ids = self.connector.get_mimic_val_ids(val_seed_patient_ids)
        test_ids = self.connector.get_mimic_test_ids(test_seed_patient_ids)
        
        
        train_dataset = GraphDataset(self.db_type, train_ids, hops_limits=[self.LIMIT, self.LIMIT], batch_size=self.BATCH_SIZE, split='MIMIC_TRAIN', use_full_batch=self.use_full_batch)
        val_dataset = GraphDataset(self.db_type, val_ids, hops_limits=[self.LIMIT, self.LIMIT], batch_size=self.BATCH_SIZE, split='MIMIC_VAL', use_full_batch=self.use_full_batch)
        test_dataset = GraphDataset(self.db_type, test_ids, hops_limits=[self.LIMIT, self.LIMIT], batch_size=self.BATCH_SIZE, split='MIMIC_TEST', use_full_batch=self.use_full_batch)

        dataloader = Dataloader(self.MAX_RAM_GB, self.BATCH_SIZE, self.NUM_WORKERS, self.hops, self.LIMIT)
        train_loader = dataloader.get_train_loader(train_dataset)
        train_loader.name = "MIMIC_TRAIN"
        val_loader = dataloader.get_val_loader(val_dataset)
        val_loader.name = "MIMIC_VAL"
        test_loader = dataloader.get_test_loader(test_dataset)
        test_loader.name = "MIMIC_TEST"

        return train_loader, val_loader, [val_loader, test_loader]

    @property
    def name(self):
        return "MIMIC"