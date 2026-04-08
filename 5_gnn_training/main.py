import torch
from torch import nn
from Neo4jDataset import Neo4jGraphDataset
from sklearn.model_selection import train_test_split
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

if __name__ == '__main__':
    
    ## Load configuration parameters    
    config = configparser.ConfigParser()
    config.read('config.ini')
    seed_everything(int(config['RANDOM']['seed']))
    feature_set_name = config['PANEL']['panel_name']
    include_sbc = config['PANEL'].getboolean('include_sbc', fallback=False)
    
    BATCH_SIZE = int(eval(config['TRAINING']['batch_size']))
    NUM_WORKERS = int(config['TRAINING']['num_workers_for_loading'])
    LIMIT = int(config['TRAINING']['limit_neighbors'])
    MAX_RAM_GB = int(config['TRAINING']['max_ram_gb'])
    
    with open("./0_mimic_preprocess/features/feature_names.txt", "r") as f:
        feature_names = f.read().replace("[", "").replace("]", "").replace("'", "").split(",")
        print(f"Using features: {feature_names} ({len(feature_names)} features)")
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hops = 2 ##rgaph depth for neighborhood sampling
    out_channels = 1
    in_channels = len(feature_names)
    
    print(f"Using BATCH_SIZE={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}, LIMIT={LIMIT}, MAX_RAM_GB={MAX_RAM_GB}, IN_CHANNELS={in_channels}, OUT_CHANNELS={out_channels}, DEVICE={device}")

    dataloader_containers = []
    sbc_training = SBC_Training(hops, MAX_RAM_GB, BATCH_SIZE, NUM_WORKERS, LIMIT)
    if sbc_training.has_sbc_nodes and include_sbc:
        dataloader_containers.append(sbc_training)
    mimic_training = Mimic_Training(hops, MAX_RAM_GB, BATCH_SIZE, NUM_WORKERS, LIMIT)
    if mimic_training.has_mimic_nodes:
        dataloader_containers.append(mimic_training)

    model_evaluation = ModelEvaluation(device = device, evaluation_fun=roc_auc_score, is_evaluation_fun_probabilistic=True)
    
        
    space = {
                'lr': hp.loguniform('lr', -3, -2),
                'weight_decay': hp.loguniform('weight_decay', -12, -4),
                'hidden_channels': hp.choice('hidden_channels', [64, 128, 256]),
                'num_layers': hp.choice('num_layers', [2, 3, 4]),
                'dropout': hp.uniform('dropout', 0.0, 0.5),
                'heads': hp.choice('heads', [2, 4, 8]),
                'activation': hp.choice('activation', [nn.ReLU(), nn.ELU(), nn.Tanh()]),
                'skip_connections': hp.choice('skip_connections', [False, True])
    }
    
    for dataloader_container in dataloader_containers:
        print(f"Processing dataset: {dataloader_container.name}")
        exp_name = f"{dataloader_container.name}_{feature_set_name}"
        

        train_loader, val_loader, test_loaders = dataloader_container.get_dataloaders()
        hyperparam_tuning_start_time = time.time()
        model_tuning = ModelTuning(device=device, model_evaluation=model_evaluation)
        ##TODO: Enable tuning
        # best_hyperparams = model_tuning.eval_hyperparameters(space, train_loader, val_loader, in_channels=in_channels, out_channels=out_channels, max_evals=20, verbosity=True)
        hyperparam_tuning_end_time = time.time() 
        # print("Best hyperparameters:", best_hyperparams)
        best_hyperparams = {
            'lr': 0.001,
            'weight_decay': 1e-5,
            'hidden_channels': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'heads': 4,
            'activation': nn.ReLU(),
            'skip_connections': False
        }
        
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
        model_training = ModelTraining(device=device, model_evaluation = model_evaluation, lr=lr, weight_decay=weight_decay, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout, heads=heads, activation=activation, skip_connections=skip_connections)
        model_training.train(train_loader, val_loader, num_epochs=100)
        training_end_time = time.time()

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(f"evaluations_{feature_set_name}")

        with mlflow.start_run():
            mlflow.set_tag("model", "GNN")
            mlflow.set_tag("approach", "Graph-based")
            mlflow.set_tag("feature_set", feature_set_name)
            mlflow.log_params(best_hyperparams)

            
            for test_loader in test_loaders:
                inference_start_time = time.time()
                test_auroc = model_evaluation.eval_model(model_training.model, test_loader)
                inference_end_time = time.time()
                print(f"EVALUATING {test_loader.name} --- AUROC: {test_auroc:.4f}")
                
                mlflow.log_metric(f"{test_loader.name}__auroc", test_auroc)
                mlflow.log_metric(f"{test_loader.name}__inference_time_seconds", inference_end_time - inference_start_time)
            mlflow.log_metric(f"{exp_name}__hyperparameter_tuning_time_seconds", hyperparam_tuning_end_time - hyperparam_tuning_start_time)
            mlflow.log_metric(f"{exp_name}__training_time_seconds", training_end_time - training_start_time)
