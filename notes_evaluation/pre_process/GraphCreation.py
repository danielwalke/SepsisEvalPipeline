import torch
import os
import regex as re
import json
import ollama
import time
import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import degree
import pandas as pd

class GraphCreation:
    def __init__(self):
        self.out_dir = "./notes_evaluation/graph_data/"
        os.makedirs(self.out_dir, exist_ok=True)
        self.statistics = {
            "avg_in_degrees": [],
            "avg_out_degrees": [],
            "max_in_degrees": [],
            "max_out_degrees": [],
            "hadm_ids": [],
            "row_ids": [],
            "labels": [],
            "num_nodes": [],
            "num_edges": [],
            "num_words": []
        }

    @staticmethod
    def identify_paragraphs(text):
        paragraph_pattern = re.compile(r'(?<=\n)([A-Z][^\n]+:)(.*?)(?=\n[A-Z][^\n]+:|\Z)', re.DOTALL)
        paragraphs = paragraph_pattern.findall(text)
        return paragraphs
    
    @staticmethod
    def identify_clinical_information_substructures(text):
        clinical_information_pattern = re.compile(r'\b\w+:\s(?:[0-9./%]+[\s,]*)+|(?u)\b\w\w+\b')
        clinical_information = clinical_information_pattern.findall(text)
        return clinical_information
    
    @staticmethod
    def embed_text(text):
        response = ollama.embed(model="mxbai-embed-large:latest", input=[text])
        embeddings = response['embeddings'][0]
        return embeddings
    
    def add_statistics(self, graph, clinical_note):
        in_degree = degree(graph.edge_index[1], num_nodes=graph.x.size(0))
        out_degree = degree(graph.edge_index[0], num_nodes=graph.x.size(0))
        self.statistics["avg_in_degrees"].append(in_degree.mean().item())
        self.statistics["avg_out_degrees"].append(out_degree.mean().item())
        self.statistics["max_in_degrees"].append(in_degree.max().item())
        self.statistics["max_out_degrees"].append(out_degree.max().item())
        self.statistics["hadm_ids"].append(graph.hadm_id)
        self.statistics["row_ids"].append(graph.row_id)
        self.statistics["labels"].append(graph.y.item())
        self.statistics["num_nodes"].append(graph.x.size(0))
        self.statistics["num_edges"].append(graph.edge_index.size(1))
        num_words = len(clinical_note.split())
        self.statistics["num_words"].append(num_words)

    def process_notes(self, path = "./notes_evaluation/batched_notes/"):
        all_files = os.listdir(path)        
        for json_file in tqdm.tqdm(all_files):
            if not json_file.endswith(".json"): continue
            

            with open(os.path.join(path, json_file), 'r') as f:
                note_data = json.load(f)
                label = note_data.get("label", "unknown")
                hadm_id = note_data.get("hadm_id", "unknown")
                row_id = note_data.get("row_id", "unknown")
                text = note_data.get("text", "")
                if len(text.strip()) == 0: continue
                if os.path.exists(os.path.join(self.out_dir, f"graph_hadm_{hadm_id}_row_{row_id}.pt")): continue
                
                paragraphs = self.identify_paragraphs(text)

                if len(paragraphs) == 0: continue
                nodes = []
                edge_index = []
                whole_note_embedding = self.embed_text(text)
                nodes.append(whole_note_embedding)
                
                
                for paragraph_tuple in paragraphs:
                    header, paragraph_text = paragraph_tuple
                    combined_paragraph = header + paragraph_text
                    ## TODO: Maybe it make sense to extend this in future for example extracting single lines with lab measurements
                    #clinical_information = self.identify_clinical_information_substructures(combined_paragraph)
                    embedded_header = self.embed_text(header)
                    embedded_paragraph = self.embed_text(paragraph_text)
                    embedded_combined = self.embed_text(combined_paragraph)
                    nodes.extend([embedded_combined, embedded_header, embedded_paragraph])
                    edge_index.append((0, len(nodes)-3))  # whole note to combined
                    edge_index.append((len(nodes)-3, len(nodes)-2))  # combined to header
                    edge_index.append((len(nodes)-3, len(nodes)-1))  # combined to paragraph
                nodes = torch.tensor(nodes, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                data = Data(x=nodes, edge_index=edge_index, y=torch.tensor([int(label)]))
                data.hadm_id = hadm_id
                data.row_id = row_id
                self.add_statistics(data, text)
                torch.save(data, os.path.join(self.out_dir, f"graph_hadm_{hadm_id}_row_{row_id}.pt"))
        return pd.DataFrame(self.statistics)
                
    
if __name__ == "__main__":
    gc = GraphCreation()
    statistics_df = gc.process_notes()
    statistics_df.to_csv("./notes_evaluation/log_information/graph_statistics.csv", index=False)
    print(statistics_df)