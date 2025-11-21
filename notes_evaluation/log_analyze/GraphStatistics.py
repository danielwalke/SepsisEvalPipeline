import pandas as pd
class GraphStatistics:
    def __init__(self):
        self.statistics_df = pd.read_csv("./notes_evaluation/log_information/graph_statistics.csv")
    
    def get_statistics(self):
        self.statistics_df.pop("max_in_degrees")
        self.statistics_df.pop("avg_in_degrees")
        return self.statistics_df
    
if __name__ == "__main__":
    gs = GraphStatistics()
    stats_df = gs.get_statistics()
    stats_df["labels"] = stats_df["labels"].astype(int)
    pos_df = stats_df[stats_df["labels"] == 1]
    neg_df = stats_df[stats_df["labels"] == 0]
    print(stats_df.describe())
    print("Positive class statistics:")
    print(pos_df.describe())
    print("Negative class statistics:")
    print(neg_df.describe())