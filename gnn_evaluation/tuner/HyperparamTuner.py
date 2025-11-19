import torch
from torch.nn import functional as F
from gnn_evaluation.model.GAT_Model import Model
from gnn_evaluation.model.ModelWrapper import ModelWrapperClassifications
from sklearn.metrics import roc_auc_score
import hyperopt
from hyperopt import fmin, tpe

class HyperparamTuner:
    def __init__(self, sbc_graphloader, device, max_evals=50):
        self.sbc_graphloader = sbc_graphloader
        self.device = device
        self.max_evals = max_evals
        self.evaluation_count = 0

    def objective(self, params):
        LR = params['lr']
        HIDDEN_DIM = int(params['hidden_dim'])
        HEADS = int(params['heads'])
        DROPOUT = params['dropout']

        model = Model(in_dim=self.sbc_graphloader.train_graph.num_features, hidden_dim=HIDDEN_DIM, edge_dim=1, out_dim=1, heads=HEADS, dropout=DROPOUT, non_lin=F.relu)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train_labels = self.sbc_graphloader.train_graph.y[self.sbc_graphloader.train_graph.train_mask]
        pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float, device=self.device))
            
        model_wrapper = ModelWrapperClassifications(model=model, model_name="GAT_SBC", epochs=5000, patience=5, pred_proba_based_metric=True, minimize_metric=False, eval_metric=roc_auc_score)
        model_wrapper.train(self.sbc_graphloader.train_loader, self.sbc_graphloader.val_loader, optimizer, criterion, self.device)
        val_auc = model_wrapper.evaluate(self.sbc_graphloader.val_loader, self.device)
        self.evaluation_count += 1
        print(f"Evaluation {self.evaluation_count}: params={params}, val_auc={val_auc}")
        return {'loss': -val_auc, 'status': hyperopt.STATUS_OK}
    
    def tune(self):
        space = {
            'lr': hyperopt.hp.loguniform('lr', -7, -3),
            'hidden_dim': hyperopt.hp.quniform('hidden_dim', 64, 256, 16),
            'heads': hyperopt.hp.quniform('heads', 2, 8, 1),
            'dropout': hyperopt.hp.quniform('dropout', 0.0, 0.5, 0.05)
        }
        best_params = fmin(self.objective, space, algo=tpe.suggest, max_evals=self.max_evals, verbose = False)
        return best_params