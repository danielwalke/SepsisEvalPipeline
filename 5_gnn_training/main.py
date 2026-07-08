import torch
from torch import nn
from connectors.Neo4jConnector import Neo4jConnector
from connectors.SQLiteConnector import SQLiteConnector
from ModelTraining import ModelTraining
from ModelEvaluation import ModelEvaluation
from sklearn.metrics import roc_auc_score
from Dataloader import Dataloader
import configparser
from ModelTuning import ModelTuning
from hyperopt import hp
import random
import os
import numpy as np
import mlflow
from SbcTraining import SBC_Training
from MimicTraining import Mimic_Training
import time

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cross_evaluate(dataloader_containers, model_evaluation, model_training):
    for dataloader_container in dataloader_containers:
        _, _, test_loaders = dataloader_container.get_dataloaders()
        for test_loader in test_loaders:
            print(f"Evaluating model trained on {dataloader_container.name} using test set: {test_loader.name}")
            inference_start_time = time.time()
            test_auroc = model_evaluation.eval_model(model_training.model, test_loader)
            inference_end_time = time.time()
            print(f"EVALUATING {test_loader.name} --- AUROC: {test_auroc:.4f}")
            
            mlflow.log_metric(f"{test_loader.name}__AUROC", test_auroc)
            mlflow.log_metric(f"{test_loader.name}__inference_time_seconds", inference_end_time - inference_start_time)

def get_connector():
    db = "sqlite"
    if db == "neo4j":
        connector = Neo4jConnector(uri="bolt://localhost:7687", user="neo4j", password="password")
    elif db == "sqlite":
         db_path = os.environ.get("DB_PATH", "/app/db_data/mimic_sbc_graph.db")
         connector = SQLiteConnector(db_path=db_path)
    else:
        raise ValueError("Unsupported database type. Please choose 'neo4j' or 'sqlite'.")
    return connector

if __name__ == '__main__':
    print("Starting GNN training and evaluation pipeline...")
    ## Load configuration parameters    
    config = configparser.ConfigParser()
    use_full_batch = False ## True doesnt work for 40gib VRAM
    print(os.listdir("/app/config/"))
    db_type = "sqlite"  
    config_path = '/app/config/config.ini'
    # config.read('/app/config/config.ini')
    
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config.read_file(f)
        
    print(f"Successfully loaded sections: {config.sections()}")
    
    # 3. Read the seed
    seed_everything(int(config['RANDOM']['seed']))

    feature_set_name = config['PANEL']['panel_name']
    include_sbc = config['PANEL'].getboolean('include_sbc', fallback=False)
    checkpoint_path = os.path.expanduser("/app/checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    connector = get_connector()
    BATCH_SIZE = 100_000_000 if use_full_batch else int(eval(config['TRAINING']['batch_size']))
    NUM_WORKERS = 3 if use_full_batch else int(config['TRAINING']['num_workers_for_loading'])
    LIMIT = int(config['TRAINING']['limit_neighbors'])
    MAX_RAM_GB = int(config['TRAINING']['max_ram_gb'])
    
    with open("/app/0_mimic_preprocess/features/feature_names.txt", "r") as f:
        feature_names = f.read().replace("[", "").replace("]", "").replace("'", "").split(",")
        print(f"Using features: {feature_names} ({len(feature_names)} features)")
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hops = 2 ##rgaph depth for neighborhood sampling
    out_channels = 1
    in_channels = len(feature_names)
    
    print(f"Using BATCH_SIZE={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}, LIMIT={LIMIT}, MAX_RAM_GB={MAX_RAM_GB}, IN_CHANNELS={in_channels}, OUT_CHANNELS={out_channels}, DEVICE={device}")

    dataloader_containers = []
    sbc_training = SBC_Training(hops, db_type, MAX_RAM_GB, BATCH_SIZE, NUM_WORKERS, LIMIT, use_full_batch=use_full_batch)
    if sbc_training.has_sbc_nodes and include_sbc:
        dataloader_containers.append(sbc_training)
    mimic_training = Mimic_Training(hops, db_type, MAX_RAM_GB, BATCH_SIZE, NUM_WORKERS, LIMIT, use_full_batch=use_full_batch)
    if mimic_training.has_mimic_nodes:
        dataloader_containers.append(mimic_training)

    model_evaluation = ModelEvaluation(device = device, evaluation_fun=roc_auc_score, is_evaluation_fun_probabilistic=True)
    
   
    space = {
                'lr': hp.loguniform('lr',np.log(0.0005), np.log(0.005)),
                'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-4)),
                'hidden_channels': hp.choice('hidden_channels', [64, 128, 256]),
                'num_layers': hp.choice('num_layers', [2, 3, 4]),
                'dropout': hp.uniform('dropout', 0.0, 0.3),
                'heads': hp.choice('heads', [2, 4, 8]),
                'activation': hp.choice('activation', [nn.ReLU(), nn.ELU()]),
                'skip_connections': hp.choice('skip_connections', [False])
    }
    
    for dataloader_container in dataloader_containers:
        print(f"Processing dataset: {dataloader_container.name}")
        exp_name = f"{dataloader_container.name}_{feature_set_name}"
        checkpoint_exp_path = os.path.join(checkpoint_path, exp_name)
        pos_weight = dataloader_container.get_pos_weight()
        os.makedirs(checkpoint_exp_path, exist_ok=True)

        train_loader, val_loader, test_loaders = dataloader_container.get_dataloaders()
        hyperparam_tuning_start_time = time.time()
        
        model_tuning = ModelTuning(device=device, model_evaluation=model_evaluation, pos_weight= pos_weight)
        best_hyperparams = model_tuning.eval_hyperparameters(space, train_loader, val_loader, in_channels=in_channels, out_channels=out_channels, max_evals=20, verbosity=True)
        hyperparam_tuning_end_time = time.time()        
        
        # best_hyperparams = {
        #     'lr': 0.001,
        #     'weight_decay': 1e-5,
        #     'hidden_channels': 128,
        #     'num_layers': 2,
        #     'dropout': 0.2,
        #     'heads': 4,
        #     'activation': nn.ReLU(),
        #     'skip_connections': False
        # }
        
        lr = best_hyperparams['lr']
        weight_decay = best_hyperparams['weight_decay']
        hidden_channels = int(best_hyperparams['hidden_channels'])
        num_layers = int(best_hyperparams['num_layers'])
        dropout = best_hyperparams['dropout']
        heads = int(best_hyperparams['heads'])
        activation = best_hyperparams['activation']
        skip_connections = best_hyperparams['skip_connections']
        
        print(f"Using lr={lr}, weight_decay={weight_decay}, hidden_channels={hidden_channels}, out_channels={out_channels}, num_layers={num_layers}, dropout={dropout}, heads={heads}, activation={activation}, skip_connections={skip_connections}")
        training_start_time = time.time()
        model_training = ModelTraining(device=device, model_evaluation = model_evaluation, pos_weight=pos_weight, lr=lr, weight_decay=weight_decay, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout, heads=heads, activation=activation, skip_connections=skip_connections)
        model_training.train(train_loader, val_loader, num_epochs=100)
        model_training.save_checkpoint(filepath=os.path.join(checkpoint_exp_path, "best_model.pth"))
        
        training_end_time = time.time()

        mlflow.set_tracking_uri("http://mlflow-server:5000")
        mlflow.set_experiment(f"evaluations_{feature_set_name}")

        with mlflow.start_run(run_name=f"GNN_{dataloader_container.name}"):
            mlflow.set_tag("model", "GNN")
            mlflow.set_tag("dataset", dataloader_container.name)
            mlflow.set_tag("approach", "Graph-based")
            mlflow.set_tag("feature_set", feature_set_name)
            mlflow.log_params(best_hyperparams)
            mlflow.log_metric(f"hyperparameter_tuning_time_seconds", hyperparam_tuning_end_time - hyperparam_tuning_start_time)
            mlflow.log_metric(f"training_time_seconds", training_end_time - training_start_time)
            cross_evaluate(dataloader_containers, model_evaluation, model_training)    
