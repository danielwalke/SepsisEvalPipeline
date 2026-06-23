from connectors.Neo4jConnector import Neo4jConnector
from connectors.SQLiteConnector import SQLiteConnector
from GraphDataset import GraphDataset
from sklearn.model_selection import train_test_split
from Dataloader import Dataloader

class SBC_Training:
    def __init__(self, hops, MAX_RAM_GB, BATCH_SIZE, NUM_WORKERS, LIMIT):
        db = "sqlite"
        self.connector = SQLiteConnector() if db == "sqlite" else Neo4jConnector()
        self.has_sbc_nodes = self.connector.has_sbc_nodes()
        self.MAX_RAM_GB = MAX_RAM_GB
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_WORKERS = NUM_WORKERS
        self.LIMIT = LIMIT
        self.hops = hops

    def get_dataloaders(self):
        self.connector.scale_and_add_pos_enc_to_features('SBC_TRAIN', 'SBC_TEST', 'SBC_EXT_TEST')

        train_seed_patient_ids = self.connector.get_sbc_patient_train_ids()
        test_seed_patient_ids = self.connector.get_sbc_patient_test_ids()
        ext_test_seed_patient_ids = self.connector.get_sbc_patient_ext_test_ids()

        train_seed_patient_ids, val_seed_patient_ids = train_test_split(train_seed_patient_ids, test_size=0.2, random_state=42)
    
        train_ids = self.connector.get_sbc_train_ids(train_seed_patient_ids)
        val_ids = self.connector.get_sbc_train_ids(val_seed_patient_ids)
        test_ids = self.connector.get_sbc_test_ids(test_seed_patient_ids)
        ext_test_ids = self.connector.get_sbc_ext_test_ids(ext_test_seed_patient_ids)
        
        
        train_dataset = GraphDataset(self.connector, train_ids, hops_limits=[self.LIMIT, self.LIMIT], batch_size=self.BATCH_SIZE, split='SBC_TRAIN')
        val_dataset = GraphDataset(self.connector, val_ids, hops_limits=[self.LIMIT, self.LIMIT], batch_size=self.BATCH_SIZE, split='SBC_TRAIN')
        test_dataset = GraphDataset(self.connector, test_ids, hops_limits=[self.LIMIT, self.LIMIT], batch_size=self.BATCH_SIZE, split='SBC_TEST')
        ext_test_dataset = GraphDataset(self.connector, ext_test_ids, hops_limits=[self.LIMIT, self.LIMIT], batch_size=self.BATCH_SIZE, split='SBC_EXT_TEST')
        
        dataloader = Dataloader(self.MAX_RAM_GB, self.BATCH_SIZE, self.NUM_WORKERS, self.hops, self.LIMIT)
        train_loader = dataloader.get_train_loader(train_dataset)
        train_loader.name = "SBC_TRAIN"
        val_loader = dataloader.get_val_loader(val_dataset)
        val_loader.name = "SBC_VAL"
        test_loader = dataloader.get_test_loader(test_dataset)
        test_loader.name = "SBC_TEST"
        ext_test_loader = dataloader.get_test_loader(ext_test_dataset)
        ext_test_loader.name = "SBC_EXT_TEST"

        return train_loader, val_loader, [val_loader, test_loader, ext_test_loader]

    @property
    def name(self):
        return "SBC"
        
        
        