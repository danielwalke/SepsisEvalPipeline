# Sepsis Evaluation pieline

Requirements:
Docker installed
./0_mimic_preproocess/icumap.csv from  https://github.com/ampel-leipzig/sbcdata/tree/main/inst/extdata/mimic-iv-1.0
./0_mimic_preproocess/d_labitems.csv from MIMIC-IV https://physionet.org/content/mimiciv/2.2/ -> But its necessary to rewrite hemoglobin to HGB, White Blood Cells to WBC, Red Blood Cells to RBC, Mean Corpuscular Volume to MCV, Plathelets to PLT
./mimic/hosp from https://physionet.org/content/mimiciv/2.2/
MLFlow


## 0. Mimic Preprocessing
We used the R scripts from Gibbs et al. (https://github.com/ampel-leipzig/sbcdata) to pre-process and label MIMIC data according to the criteria defined in Steinbach et al. [1]. This pre-processing was enwrapped in a docker container to facilitating usage and the script was adopted to consider various feature sets based on a specific set of feature item ids (process_files.R). Additionally, we investigated which features are often measured together as a set (e.g., complete blood cound data, basic metabolic panel, liver panel) using mimic_lab_frequency. Based on information in the config.ini feature sets separated by "_" can be combined usin the panel_name_to_feature_codes.py to extract all itemids that can then be used as input for the preprocess_files.R preprocessing script.
Requirements:
./0_mimic_preproocess/icumap.csv from  https://github.com/ampel-leipzig/sbcdata/tree/main/inst/extdata/mimic-iv-1.0
./0_mimic_preproocess/d_labitems.csv from MIMIC-IV https://physionet.org/content/mimiciv/2.2/ -> But its necessary to rewrite hemoglobin to HGB, White Blood Cells to WBC, Red Blood Cells to RBC, Mean Corpuscular Volume to MCV, Plathelets to PLT
./mimic/hosp from https://physionet.org/content/mimiciv/2.2/

Output: 
Folders in:
./0_mimic_preproocess/preprocessed_file


## 1. Pre-processing
Then, we preprocessed the SBC data and MIMIC data according to the pre-processing described by Steinbach et al. [1]. 
Input:
./0_mimic_preproocess/preprocessed_file

Output:
./1_preprocess/data/preprocessed_data



## 2. Machine learning baselines
Then, we developed machine learning baselines with these feature sets with a simple Logistic Regression and a Random Forest.
Input:
./1_preprocess/data/preprocessed_data

Output:
- AUROC scores in MLFlow localhost:5000
- Trained models in 2_baseline/models

## 3. Graph construction
In the third step, we developed directed, patient-dentric graphs according to Walke et al. [2]. However, we added an edge weight based on the difference of the current measurement to previous measurements between zero (measurements longer ago) and one (closer to the current measurement).

Input:
./1_preprocess/data/preprocessed_data

Output: 
./3_graph_construction/data

## 4. Database upload
Then, we load the constructed graphs stored as CSV files in a Neo4j database, edge weights are stored as property on the edges.
Input:
./3_graph_construction/data

Output:
./4_db_upload/neo4j_data

## 5. GNN training
Small mini batches based on Walke et al. [TODO: DBEval paper] are fetched and use to mini-batch train a GNN. Specifically, we used graph attention networks that can also incorporate the edge weights (i.e., time-series differences) in their data.
Input:
./4_db_upload/neo4j_data

Output:
- AUROC scores in MLFlow localhost:5000
- Trained Checkpoints in 5_gnn_training/checkpoints

## 6. GraphAware training
Again small-mini batches are used from the created graph database, but as input for the interpretable GraphAware framework from Walke et al. [3]. Here we use the difference to the mean of neighboring nodes as permutation-invariant aggregation  function and incorporate the edge-weights in the aggregation process and the XGBoostClassifier which allows a batch-wise training process. The trained XGBoost classifier can then be anaylzed with the SHAP framework from Lundberg et al [4] to investigate how the model makes its decisions.

Input: 
Input:
./4_db_upload/neo4j_data

Output:
- AUROC scores in MLFlow localhost:5000
- Trained models in 6_graph_aware/models
- Figrues for global SHAP values for increased interpretability


## 7. Access - Open TODO
MCP-Servers with GraphAware only for now -> Append to neo4j database for continously learning? Whenever we have 500 newly added nodes train the xgboost model again?

TODO: Notes integration? with embeddings? might be too much?
PROOBABLY:
We could create embeddings for notes and learn a gnn with best end to end approach and use this as sota value which is hard to interprret
But we comapre it against GraphAware with a simple bag of words approach after lemmatization of notes
-> 2nd more interpretable but first might be more powerful?
-> Would take me at leats one month
-> Might make sense to discuss my current points till now with david at least with an end-to-end trainable neo4j graph learning pipeline for sepsis prediction with re-training or re-fit potential and MCP for increased accessibility 
-> Then potentially Integration of notes?or might be overkill since costs again some months of implementation -> probably need to ask robert since i am mainly working on MCP project now

## References
[1] Applying Machine Learning to Blood Count Data Predicts Sepsis with ICU Admission. D. Steinbach, P. C. Ahrens, M. Schmidt, M. Federbusch, L. Heuft, Ch. Lübbert, M. Nauck, M. Gründling, B. Isermann, S. Gibb, Th. Kaiser 2024. Clinical Chemistry. DOI: 10.1093/clinchem/hvae001.
[2] Edges are all you need: Potential of medical time series analysis on complete blood count data with graph neural networks
Walke D, Steinbach D, Gibb S, Kaiser T, Saake G, et al. (2025) Edges are all you need: Potential of medical time series analysis on complete blood count data with graph neural networks. PLOS ONE 20(7): e0327636. https://doi.org/10.1371/journal.pone.0327636
[3] GraphAware: Interpretable machine learning on graphs; Daniel Walke, Daniel Steinbach, Alexander Schönhuth et al., 25 September 2025, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-7471432/v1]
[4] A Unified Approach to Interpreting Model Predictions. S. M. Lundberg, S. Lee 2017. Proceedings of the 31st International Conference on Neural Information Processing Systems. DOI: 10.5555/3295222.3295230.