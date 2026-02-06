from GNN_Model import GNNModel
from ModelEvaluation import ModelEvaluation
import torch
from torch import nn

class ModelTraining:
    def __init__(self, device, model_evaluation:ModelEvaluation, lr, weight_decay, in_channels, hidden_channels=128, out_channels=1, num_layers=3, dropout=0.3, heads=4, activation=nn.ReLU(), skip_connections=False):
        assert isinstance(model_evaluation, ModelEvaluation), "model_evaluation must be an instance of the ModelEvaluation class."
        assert isinstance(device, torch.device), "device must be a torch.device object indicating the device to run training on."
        assert isinstance(lr, float) and lr > 0, "Learning rate must be a positive float."
        assert isinstance(weight_decay, float) and weight_decay >= 0, "Weight decay must be a non-negative float."
        assert isinstance(in_channels, int) and in_channels > 0, "Input channels must be a positive integer."
        assert isinstance(hidden_channels, int) and hidden_channels > 0, "Hidden channels must be a positive integer."
        assert isinstance(out_channels, int) and out_channels > 0, "Output channels must be a positive integer."
        assert isinstance(num_layers, int) and num_layers >= 2, "Number of layers must be an integer greater than or equal to 2."
        assert isinstance(dropout, float) and 0 <= dropout < 1, "Dropout must be a float in the range [0, 1)."
        assert isinstance(heads, int) and heads > 0, "Heads must be a positive integer."
        assert isinstance(activation, nn.Module), "Activation must be a PyTorch activation function (nn.Module)."
        assert isinstance(skip_connections, bool), "Skip connections must be a boolean value."
        
        self.device = device
        self.model_evaluation = model_evaluation
        self.model = GNNModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout, heads=heads, activation=activation, skip_connections=skip_connections).to(self.device)
        pos_weight = torch.tensor([664.0]).to(self.device) #525.0 for mimic, 664.0 for sbc
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.best_val_auroc = float('-inf')
        self.PATIENCE = 10
        self.patience_counter = 0
        self.best_model_state_dict = None
        
    def train(self, train_loader, val_loader, num_epochs=100):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                if batch is None: continue
                
                batch = batch.to(self.device)
                
                
                out = self.model(batch.x, batch.edge_index, batch.edge_attr)
                
                target = batch.y[batch.batch_mask]
                logits = out[batch.batch_mask]
                
                loss = self.criterion(logits, target)
                self.optimizer.zero_grad()
                loss.backward()            
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()

            print(f"Epoch {epoch} | Loss: {total_loss:.4f}")
            if epoch % 1 == 0:
                val_auroc = self.model_evaluation.eval_model(self.model, val_loader)
                print(f"--- VAL AUROC: {val_auroc:.4f} ---")
                if val_auroc > self.best_val_auroc:
                    self.best_val_auroc = val_auroc
                    self.best_model_state_dict = self.model.state_dict()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                if self.patience_counter >= self.PATIENCE: ## TODO might prefer mean val_auroc over last few epochs instead of single epoch val_auroc to determine early stopping
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
        self.model.load_state_dict(self.best_model_state_dict)
        return self.model
                