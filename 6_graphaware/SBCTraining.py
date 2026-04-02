from NodeSplitContainer import NodeSplitContainer, NodeSplitInformation
from sklearn.model_selection import train_test_split

class SBCTraining:
    def __init__(self):
        self.train_label_name = "SBC_TRAIN"
        self.val_label_name = "SBC_TRAIN"
        self.test_label_name = "SBC_TEST"
        self.ext_test_label_name = "SBC_EXT_TEST"
        pass
    
    def get_node_split_containers(self, connector):
        all_train_patient_ids = connector.get_patient_ids('SBC_TRAIN')
        train_patient_ids, val_patient_ids = train_test_split(all_train_patient_ids, test_size=0.2, random_state=42)
        train_seed_ids = connector.get_sbc_train_ids(train_patient_ids)
        val_seed_ids = connector.get_sbc_val_ids(val_patient_ids)

        self.train_split_info = NodeSplitInformation(train_seed_ids, self.train_label_name, "WHERE n.id IN $ids", "SBC_TRAIN")
        self.val_split_info = NodeSplitInformation(val_seed_ids, self.val_label_name, "WHERE n.id IN $ids", "SBC_VAL")
        self.test_split_info = NodeSplitInformation(connector.get_ids('SBC_TEST'), self.test_label_name, "", "SBC_TEST")
        self.ext_test_split_info = NodeSplitInformation(connector.get_ids('SBC_EXT_TEST'), self.ext_test_label_name, "", "SBC_EXT_TEST")
        return NodeSplitContainer(self.train_split_info, self.val_split_info, self.test_split_info, self.ext_test_split_info)