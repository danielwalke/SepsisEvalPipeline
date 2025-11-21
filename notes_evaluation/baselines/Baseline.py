import json
from sklearn.model_selection import train_test_split
import os
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from notes_evaluation.train.Notes import NotesEvaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.ensemble import RandomForestClassifier

class Baseline:
    def __init__(self, ModelClass = LogisticRegression):
        self.notes_dir = "./notes_evaluation/batched_notes/"
        self.train_row_ids = []
        self.val_row_ids = []
        self.test_row_ids = []
        self.train_notes = []
        self.val_notes = []
        self.test_notes = []
        self.vectorizer = None
        self.train_X = None
        self.val_X = None
        self.test_X = None
        self.train_labels = []
        self.val_labels = []
        self.test_labels = []
        self.model = None
        self.ModelClass = ModelClass

    def get_graph_splits(self):
        ne = NotesEvaluation()
        ne.load_graphs()
        ne.split_graphs()
        self.train_row_ids = [graph.row_id for graph in ne.train_graphs]
        self.val_row_ids = [graph.row_id for graph in ne.val_graphs]
        self.test_row_ids = [graph.row_id for graph in ne.test_graphs]          

    def load_notes(self):
        print("Loading notes...")
        for file in os.listdir(self.notes_dir):
            if not file.endswith(".json"): continue
            with open(os.path.join(self.notes_dir, file), 'r') as f:
                notes = json.load(f)
                row_id = notes['row_id']
                if row_id in self.train_row_ids:
                    self.train_notes.append(notes)
                    self.train_labels.append(int(notes['label']))
                elif row_id in self.val_row_ids:
                    self.val_notes.append(notes)
                    self.val_labels.append(int(notes['label']))
                elif row_id in self.test_row_ids:
                    self.test_notes.append(notes)
                    self.test_labels.append(int(notes['label']))

    def create_features(self):
        print("Creating features...")
        train_texts = [note['text'] for note in self.train_notes]
        val_texts = [note['text'] for note in self.val_notes]
        test_texts = [note['text'] for note in self.test_notes]
        self.vectorizer = TfidfVectorizer() ## TODO: max_features?
        self.train_X = self.vectorizer.fit_transform(train_texts)
        self.val_X = self.vectorizer.transform(val_texts)
        self.test_X = self.vectorizer.transform(test_texts)

    def train(self, **params):
        print(f"Training {self.ModelClass.__name__} model...")
        self.model = self.ModelClass(**params)
        self.model.fit(self.train_X, self.train_labels)

    def check_is_model_trained(self):
        if self.model is None:
            raise Exception("Model not trained yet.")

    def evaluate_val(self):
        self.check_is_model_trained()
        val_probs = self.model.predict_proba(self.val_X)[:, 1]
        auc = roc_auc_score(self.val_labels, val_probs)
        return auc
    
    def evaluate_test(self):
        self.check_is_model_trained()
        test_probs = self.model.predict_proba(self.test_X)[:, 1]
        auc = roc_auc_score(self.test_labels, test_probs)
        return auc
    
    def hyperparameter_tuning(self, search_space, max_evals=50):
        print("Starting hyperparameter tuning...")
        def objective(params):
            self.train(**params)
            val_auc = self.evaluate_val()
            return {'loss': -val_auc, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials,
                    verbose=1)
        return space_eval(search_space, best)
    
if __name__ == "__main__":
    # search_space = {
    #         'C': hp.loguniform('C', -4, 4),
    #         'max_iter': hp.choice('max_iter', [100, 500, 1000, 2000])
    # }
    # ModelClass = LogisticRegression
    ModelClass = RandomForestClassifier
    search_space = {
            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
            'max_depth': hp.choice('max_depth', [None, 10, 20, 30]),
            'min_samples_split': hp.choice('min_samples_split', [2, 5, 10])
    }

    lr_baseline = Baseline(ModelClass=ModelClass)
    lr_baseline.get_graph_splits()
    lr_baseline.load_notes()
    lr_baseline.create_features()
    best_params = lr_baseline.hyperparameter_tuning(search_space, max_evals=50)
    print(f"Best hyperparameters: {best_params}")
    lr_baseline.train(**best_params)
    val_auc = lr_baseline.evaluate_val()
    test_auc = lr_baseline.evaluate_test()
    print(f"Validation AUC: {val_auc}")
    print(f"Test AUC: {test_auc}")