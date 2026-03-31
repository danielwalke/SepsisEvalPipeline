from hyperopt import fmin, tpe, STATUS_OK, Trials, space_eval
import torch
from ModelEvaluation import ModelEvaluation
from ModelTraining import ModelTraining

class ModelTuning:
    def __init__(self, device, model_evaluation: ModelEvaluation):
        assert isinstance(model_evaluation, ModelEvaluation)
        assert isinstance(device, torch.device)
        
        self.device = device
        self.model_evaluation = model_evaluation
        
    def eval_hyperparameters(self, space, train_loader, val_loader, in_channels, out_channels, max_evals=20, verbosity=True):
        ## TODO seed init
        def objective(params):
            lr = float(params['lr'])
            weight_decay = float(params['weight_decay'])
            hidden_channels = int(params['hidden_channels'])
            num_layers = int(params['num_layers'])
            dropout = float(params['dropout'])
            heads = int(params['heads'])
            activation = params['activation']
            skip_connections = bool(params['skip_connections'])

            if verbosity:
                print(f"Testing: lr={lr:.5f}, layers={num_layers}, hidden={hidden_channels}, heads={heads}")
                
            print(f"Evaluating hyperparameters: lr={lr}, weight_decay={weight_decay}, hidden_channels={hidden_channels}, num_layers={num_layers}, dropout={dropout}, heads={heads}, activation={activation}, skip_connections={skip_connections}")

            model_training = ModelTraining(
                device=self.device, 
                model_evaluation=self.model_evaluation, 
                lr=lr, 
                weight_decay=weight_decay, 
                in_channels=in_channels, 
                hidden_channels=hidden_channels, 
                out_channels=out_channels, 
                num_layers=num_layers, 
                dropout=dropout, 
                heads=heads, 
                activation=activation, 
                skip_connections=skip_connections
            )
            
            model_training.train(train_loader, val_loader)
            val_metric = self.model_evaluation.eval_model(model_training.model, val_loader)
            
            loss = -val_metric if self.model_evaluation.is_higher_metric_better else val_metric
            
            if verbosity:
                print(f"Validation METRIC: {val_metric:.4f}")

            return {'loss': loss, 'status': STATUS_OK, 'metric': val_metric}

        trials = Trials()
        
        best_indices = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=verbosity
        )

        best_hyperparams = space_eval(space, best_indices)
        
        if verbosity:
            print(f"Best Hyperparameters: {best_hyperparams}")
            best_trial_loss = min(t['result']['loss'] for t in trials.trials)
            best_metric = -best_trial_loss if self.model_evaluation.is_higher_metric_better else best_trial_loss
            print(f"Best Validation METRIC: {best_metric:.4f}")

        return best_hyperparams