from gnn_evaluation.model.GAT_Model import Model
from gnn_evaluation.model.ModelWrapper import ModelWrapperClassifications
from sklearn.metrics import roc_auc_score
from gnn_evaluation.graphloader.Mimic_Graphloader import GraphLoaderMimic
from gnn_evaluation.utils.SeedInitialization import initialize_seed
from gnn_evaluation.tuner.HyperparamTuner import HyperparamTuner
import torch
from torch.nn import functional as F

if __name__ == "__main__":
    BATCH_SIZE = 512
    LR= 1e-3
    HIDDEN_DIM = 128
    HEADS = 8
    DROPOUT = 0.0
    
    initialize_seed(42)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mimic_graphloader = GraphLoaderMimic()
    mimic_graphloader.initialize(BATCH_SIZE)

    hyperparam_tuner = HyperparamTuner(mimic_graphloader, device, max_evals=50)
    best_params = hyperparam_tuner.tune()
    print("Best Hyperparameters:", best_params)
    best_params = {
        'lr': LR,
        'hidden_dim': 128.0,
        'heads': 8.0,
        'dropout': 0.0
    }

    model = Model(in_dim=mimic_graphloader.train_graph.num_features, hidden_dim=int(best_params['hidden_dim']), edge_dim=1, out_dim=1, heads=int(best_params['heads']), dropout=best_params['dropout'], non_lin=F.relu)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    train_labels = mimic_graphloader.train_graph.y[mimic_graphloader.train_graph.train_mask]
    pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float, device=device))
        
    model_wrapper = ModelWrapperClassifications(model=model, model_name="GAT_Mimic", epochs=5000, patience=5, pred_proba_based_metric=True, minimize_metric=False, eval_metric=roc_auc_score)
    model_wrapper.train(mimic_graphloader.train_loader, mimic_graphloader.val_loader, optimizer, criterion, device)
    test_auc = model_wrapper.evaluate(mimic_graphloader.test_loader, device)
    print(f"Test AUC: {test_auc:.4f}")
