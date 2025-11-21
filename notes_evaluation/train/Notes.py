import os
import torch
from torch_geometric.utils import degree

class NotesEvaluation:
    def __init__(self):
        self.graph_data_path = "./notes_evaluation/graph_data/"
        self.graphs = []

    def load_graphs(self):
        for file in os.listdir(self.graph_data_path):
            if not file.endswith(".pt"): continue
            graph = torch.load(os.path.join(self.graph_data_path, file), weights_only=False)
            degree_start_node = degree(graph.edge_index[0])
            degree_end_node = degree(graph.edge_index[1])
            print(f"Graph: {file}")
            print(f"Degree of start node: {degree_start_node}")
            print(f"Degree of end node: {degree_end_node}")
            self.graphs.append(graph)

if __name__ == "__main__":
    ne = NotesEvaluation()
    ne.load_graphs()