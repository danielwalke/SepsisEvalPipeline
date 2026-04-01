import pandas as pd

RESULTS_LOG_PATH = "results_log.csv"
columns_heads = ["dataset", "features", "model", "approach", "test_score", "ext_test_score", "hyperparameters"]

def read_results_log():
    try:
        return pd.read_csv(RESULTS_LOG_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=columns_heads)

def write_results_log(results_log):
    results_log.to_csv(RESULTS_LOG_PATH, index=False)

def append_result(dataset, features, model, approach, test_score, hyperparameters, ext_test_score = None):
    results_log = read_results_log()
    new_entry = {
        "dataset": dataset,
        "features": str(features),
        "model": model,
        "approach": approach,
        "test_score": test_score,
        "ext_test_score": ext_test_score,
        "hyperparameters": str(hyperparameters)
    }
    results_log = results_log.append(new_entry, ignore_index=True)
    write_results_log(results_log)
    return results_log          

