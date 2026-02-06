from torch.utils.data import DataLoader
from utils.MemoryCalculation import calculate_loader_params

class Dataloader:
    def __init__(self, max_gb, batch_size, num_workers, hops, limit):
        assert max_gb > 0 and isinstance(max_gb, (int, float)), "max_gb must be greater than 0 and a number"
        assert batch_size > 0 and isinstance(batch_size, int), "batch_size must be greater than 0 and an integer"
        assert num_workers >= 0 and isinstance(num_workers, int), "num_workers must be a non-negative integer"
        assert hops >= 2 and isinstance(hops, int), "hops must be an integer and at least 2"
        assert limit >= 0 and isinstance(limit, int), "limit must be a non-negative integer for neighbor sampling"
        
        self.max_gb = max_gb
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hops = hops
        self.limit = limit
        
        max_gb_train = self.max_gb * .6  # 60% for training
        max_gb_val = self.max_gb * .2    # 20% for validation
        max_gb_test = self.max_gb * .2   # 20% for testing
        self.prefetch_train = calculate_loader_params(max_gb_train, self.batch_size, self.num_workers, self.hops, self.limit)
        self.prefetch_val = calculate_loader_params(max_gb_val, self.batch_size, self.num_workers, self.hops, self.limit)
        self.prefetch_test = calculate_loader_params(max_gb_test, self.batch_size, self.num_workers, self.hops, self.limit)
    
    def get_train_loader(self, train_dataset):
        return DataLoader(
            train_dataset, 
            batch_size=None,  
            shuffle=True,     
            num_workers=self.num_workers-2,
            prefetch_factor=self.prefetch_train, 
        persistent_workers=True
    )  
    
    def get_val_loader(self, val_dataset):
        return DataLoader(
            val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=2,
            prefetch_factor=self.prefetch_val,
            persistent_workers=True
        )
    
    def get_test_loader(self, test_dataset):
        return DataLoader(
            test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=2,
        prefetch_factor=self.prefetch_test,
        persistent_workers=True
    )