
class NodeSplitInformation:
    def __init__(self, node_ids, label_name, condition, name):
        self.node_ids = node_ids
        self.label_name = label_name
        self.condition = condition
        self.name = name


class NodeSplitContainer:
    def __init__(self, train_split_information, val_split_information, *test_split_information_list):
        self.train_split_information = train_split_information
        self.val_split_information = val_split_information
        self.test_split_information_list = [val_split_information] + list(test_split_information_list)