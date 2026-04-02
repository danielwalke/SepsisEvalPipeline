import xgboost as xgb

class Neo4jDataIter(xgb.DataIter):
    def __init__(self, connector, node_label, condition, framework, batch_size=100000, target_ids=None):
        self.connector = connector
        self.node_label = node_label
        self.condition = condition
        self.framework = framework
        self.batch_size = batch_size
        self.target_ids = target_ids
        self.skip = 0
        super().__init__()

    def reset(self):
        self.skip = 0

    def next(self, input_data):
        X, y = self.connector.fetch_data_batch(
            self.node_label, 
            self.condition, 
            self.skip, 
            self.batch_size,
            self.framework,
            self.target_ids
        )
        
        if len(y) == 0:
            return 0
            
        input_data(data=X, label=y)
        self.skip += self.batch_size
        return 1
