
from training_containers.NodeSplitContainer import NodeSplitContainer, NodeSplitInformation

class MIMICTraining:
    def __init__(self):
        self.train_label_name = "MIMIC_TRAIN"
        self.val_label_name = "MIMIC_VAL"
        self.test_label_name = "MIMIC_TEST"
    
    def get_node_split_containers(self, connector):
        self.train_split_info = NodeSplitInformation(connector.get_ids('MIMIC_TRAIN'), self.train_label_name, "", "MIMIC_TRAIN")
        self.val_split_info = NodeSplitInformation(connector.get_ids('MIMIC_VAL'), self.val_label_name, "", "MIMIC_VAL")
        self.test_split_info = NodeSplitInformation(connector.get_ids('MIMIC_TEST'), self.test_label_name, "", "MIMIC_TEST")
        return NodeSplitContainer(self.train_split_info, self.val_split_info, self.test_split_info)

    @property
    def name(self):
        return "MIMIC"