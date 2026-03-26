import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lab_panels import LAB_PANELS

counts_file_path = "/Users/danielwalke/git/SepsisEvalPipeline/mimic/hosp/itemid_counts.csv"
labevents_file_path = "/Users/danielwalke/git/SepsisEvalPipeline/mimic/hosp/labevents.csv"
d_labevents_file_path = "/Users/danielwalke/git/SepsisEvalPipeline/mimic/hosp/d_labitems.csv"
if os.path.exists(counts_file_path):
    itemid_counts_df = pd.read_csv(counts_file_path)
    print("Loaded from existing counts file")
else:
    labevents_df = pd.read_csv(labevents_file_path)
    itemid_counts_df = labevents_df["itemid"].value_counts().reset_index()
    itemid_counts_df.columns = ["itemid", "count"]
    itemid_counts_df.to_csv(counts_file_path, index=False)
    print("Calculated and exported to csv")

d_labitems_df = pd.read_csv(d_labevents_file_path)
labeled_itemid_counts = pd.merge(d_labitems_df, itemid_counts_df, how = "right", on = "itemid").iloc[:70]
labeled_itemid_counts["label"] = labeled_itemid_counts["label"] +" (" + labeled_itemid_counts["fluid"] + ")"

def panel_view(lab_list, panel_name):
    filtered_df = labeled_itemid_counts[labeled_itemid_counts["label"].isin(lab_list)]
    plt.figure(figsize=(10, 6))
    sns.barplot(x='label', y='count', data=filtered_df)
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title(f'Bar Plot of {panel_name}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{panel_name}.png')


for panel_name, lab_list in LAB_PANELS.items():
    panel_view(lab_list, panel_name)
