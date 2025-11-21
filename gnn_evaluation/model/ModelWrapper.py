import torch
from sklearn.metrics import roc_auc_score

class ModelWrapperClassifications:
    def __init__(self, model, model_name: str, epochs, patience, eval_metric = roc_auc_score, minimize_metric: bool = False, pred_proba_based_metric: bool = True):
        self.model = model
        self.model_name = model_name
        self.epochs = epochs
        self.eval_metric = eval_metric
        self.minimize_metric = minimize_metric
        self.pred_proba_based_metric = pred_proba_based_metric
        self.best_model_state = None
        self.best_metric_value = float('inf') if minimize_metric else float('-inf')
        self.patience = patience
        self.counter = 0

    def train(self, train_loader, val_loader, optimizer, criterion, device):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion, device)
            val_metric = self.evaluate(val_loader, device)
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Metric: {val_metric:.4f}")

            if (self.minimize_metric and val_metric < self.best_metric_value) or (not self.minimize_metric and val_metric > self.best_metric_value):
                self.best_metric_value = val_metric
                self.best_model_state = self.model.state_dict()
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
    def train_epoch(self, data_loader, optimizer, criterion, device):
        self.model.train()
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, edge_weight=data.edge_attr, batch = data.batch)
            if hasattr(data, 'train_mask'):
                out = out[data.train_mask]
                gt = data.y[data.train_mask]
            else:
                gt = data.y
            loss = criterion(out, gt.float())
            loss.backward()
            optimizer.step()
        return loss.item()
    
    def evaluate(self, data_loader, device):
        self.model.eval()
        with torch.no_grad():
            labels = []
            pred_probas = []
            for data in data_loader:
                data = data.to(device)
                out = self.model(data.x, data.edge_index, edge_weight=data.edge_attr, batch=data.batch)
                if hasattr(data, 'test_mask'):
                    out = out[data.test_mask]
                pred_probas.extend(torch.sigmoid(out).cpu().numpy())
                if hasattr(data, 'test_mask'):
                    labels.extend(data.y[data.test_mask].cpu().numpy())
                else:
                    labels.extend(data.y.cpu().numpy())
            if self.pred_proba_based_metric:
                metric_value = self.eval_metric(labels, pred_probas)
            else:
                metric_value = self.eval_metric(labels, [1 if p > 0.5 else 0 for p in pred_probas])
        return metric_value