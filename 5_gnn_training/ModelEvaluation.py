import torch
from sklearn.metrics import roc_auc_score


class ModelEvaluation:
    def __init__(self, device, evaluation_fun = roc_auc_score, is_evaluation_fun_probabilistic=True, is_higher_metric_better=True):
        assert callable(evaluation_fun), "evaluation_fun must be a callable function that takes (labels, probs) as input and returns a scalar metric."        
        assert isinstance(is_evaluation_fun_probabilistic, bool), "is_evaluation_fun_probabilistic must be a boolean value indicating whether the evaluation function expects probabilities or binary predictions."
        assert isinstance(device, torch.device), "device must be a torch.device object indicating the device to run evaluation on."
        assert isinstance(is_higher_metric_better, bool), "is_higher_metric_better must be a boolean value indicating whether higher values of the metric indicate better performance."
        self.device = device
        self.evaluation_fun = evaluation_fun
        self.is_evaluation_fun_probabilistic = is_evaluation_fun_probabilistic
        self.is_higher_metric_better = is_higher_metric_better  
    
    def eval_model(self, model, loader):
        model.eval()
        all_labels = []
        all_probs = []
        
        with torch.inference_mode():
            for batch in loader:
                if batch.x is None or batch.batch_mask.sum() == 0: 
                    continue
                
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                target = batch.y[batch.batch_mask].squeeze()
                logits = out[batch.batch_mask].squeeze()
                probs = torch.sigmoid(logits)
                
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        if self.is_evaluation_fun_probabilistic:
            metric_value = self.evaluation_fun(all_labels, all_probs)
        else:
            binary_preds = [1 if p >= 0.5 else 0 for p in all_probs]
            metric_value = self.evaluation_fun(all_labels, binary_preds)
        return metric_value
