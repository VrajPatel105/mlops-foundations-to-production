# 5. ml pipeline : model evaluation final stage for ml pipeline

import os
import json
import pandas as pd
import pickle
import logging

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(model_path: str):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug('Model loaded from %s', model_path)
        return model
    except FileNotFoundError as e:
        logger.error('Model file not found: %s', e)
        raise
    except pickle.UnpicklingError as e:
        logger.error('Error unpickling model: %s', e)
        raise


def load_data(test_path: str):
    """Load the featurized test data."""
    try:
        test_data = pd.read_csv(test_path)
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        logger.debug('Test data loaded, shape: %s', test_data.shape)
        return X_test, y_test
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('Empty CSV file: %s', e)
        raise


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate the model and return a metrics dict."""
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        logger.debug('Model evaluation completed: %s', metrics_dict)
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics_dict: dict, output_dir: str):
    """Save the metrics dict to a json file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.debug('Metrics saved to %s', output_dir)
    except Exception as e:
        logger.error('Error saving metrics: %s', e)
        raise


def main():
    try:
        output_dir = "1. data-versioning"
        model = load_model(os.path.join(output_dir, "model.pkl"))
        X_test, y_test = load_data("./data/features/test_bow.csv")
        metrics_dict = evaluate_model(model, X_test, y_test)
        save_metrics(metrics_dict, output_dir)
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()