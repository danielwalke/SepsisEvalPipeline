from gnn_evaluation.model.GAT_Model import Model
from gnn_evaluation.model.ModelWrapper import ModelWrapperClassifications
from sklearn.metrics import roc_auc_score
from gnn_evaluation.graphloader.Sbc_Graphloader import GraphLoaderSBC
from gnn_evaluation.utils.SeedInitialization import initialize_seed
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
    sbc_graphloader = GraphLoaderSBC()
    sbc_graphloader.initialize(BATCH_SIZE)

    model = Model(in_dim=sbc_graphloader.train_graph.num_features, hidden_dim=HIDDEN_DIM, edge_dim=1, out_dim=1, heads=HEADS, dropout=DROPOUT, non_lin=F.relu)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_labels = sbc_graphloader.train_graph.y[sbc_graphloader.train_graph.train_mask]
    pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float, device=device))
        
    model_wrapper = ModelWrapperClassifications(model=model, model_name="GAT_SBC", epochs=2, patience=5, pred_proba_based_metric=True, minimize_metric=False, eval_metric=roc_auc_score)
    model_wrapper.train(sbc_graphloader.train_loader, sbc_graphloader.val_loader, optimizer, criterion, device)
    test_auc = model_wrapper.evaluate(sbc_graphloader.test_loader, device)
    print(f"Test AUC: {test_auc:.4f}")
