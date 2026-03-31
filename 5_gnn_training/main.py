import torch
from torch import nn
from connectors.Neo4jConnector import Neo4jConnector
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
    seed_everything(42)
    
    BATCH_SIZE = int(eval(config['TRAINING']['batch_size']))
    NUM_WORKERS = int(config['TRAINING']['num_workers_for_loading'])
    LIMIT = int(config['TRAINING']['limit_neighbors'])
    MAX_RAM_GB = int(config['TRAINING']['max_ram_gb'])
    
    print(f"Using BATCH_SIZE={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}, LIMIT={LIMIT}, MAX_RAM_GB={MAX_RAM_GB}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hops = 2
    out_channels = 1
    in_channels = 7
    
    neo4j_connector = Neo4jConnector()
    neo4j_connector.scale_and_add_pos_enc_to_features('SBC_TRAIN', 'SBC_TEST', 'SBC_EXT_TEST')
    train_seed_ids = neo4j_connector.get_train_ids()
    test_seed_ids = neo4j_connector.get_test_ids()
    ## TODO We want patient ids here to usse inductive graph learning on val instead of transductive learning.
    train_seed_ids, val_seed_ids = train_test_split(train_seed_ids, test_size=0.2, random_state=42)
    
    
    train_dataset = Neo4jGraphDataset(train_seed_ids, hops_limits=[LIMIT, LIMIT], batch_size=BATCH_SIZE, split='SBC_TRAIN')
    val_dataset = Neo4jGraphDataset(val_seed_ids, hops_limits=[LIMIT, LIMIT], batch_size=BATCH_SIZE, split='SBC_TRAIN')
    test_dataset = Neo4jGraphDataset(test_seed_ids, hops_limits=[LIMIT, LIMIT], batch_size=BATCH_SIZE, split='SBC_TEST')
    
    dataloader = Dataloader(MAX_RAM_GB, BATCH_SIZE, NUM_WORKERS, hops, LIMIT)
    train_loader = dataloader.get_train_loader(train_dataset)
    val_loader = dataloader.get_val_loader(val_dataset)
    test_loader = dataloader.get_test_loader(test_dataset)
    
    
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
    model_tuning = ModelTuning(device=device, model_evaluation=model_evaluation)
    # best_hyperparams = model_tuning.eval_hyperparameters(space, train_loader, val_loader, in_channels=in_channels, out_channels=out_channels, max_evals=20, verbosity=True)
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
    model_training = ModelTraining(device=device, model_evaluation = model_evaluation, lr=lr, weight_decay=weight_decay, in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout, heads=heads, activation=activation, skip_connections=skip_connections)
    model_training.train(train_loader, val_loader, num_epochs=100)
    
    val_auroc = model_evaluation.eval_model(model_training.model, val_loader)
    test_auroc = model_evaluation.eval_model(model_training.model, test_loader)
    print(f"FINAL VAL AUROC: {val_auroc:.4f} --- FINAL TEST AUROC: {test_auroc:.4f} ---")
