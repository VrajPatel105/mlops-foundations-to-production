# 4. ml pipeline : model building

import os
import pandas as pd
import logging

import xgboost as xgb
import pickle
import yaml

logger = logging.getLogger('model_building')
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


def load_params(params_path: str) -> dict:
    """Load model building params from the params yaml file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)['model_building']
        logger.debug('Model building params retrieved')
        return params
    except FileNotFoundError as e:
        logger.error('Params file not found: %s', e)
        raise
    except KeyError as e:
        logger.error('Missing key in params file: %s', e)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file: %s', e)
        raise


def load_data(train_path: str):
    """Load the featurized training data."""
    try:
        train_data = pd.read_csv(train_path)
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        logger.debug('Training data loaded, shape: %s', train_data.shape)
        return X_train, y_train
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('Empty CSV file: %s', e)
        raise


def train_model(X_train, y_train, params: dict):
    """Train the XGBoost model."""
    try:
        xgb_model = xgb.XGBClassifier(
            eval_metric=params['eval_metric'],
            n_estimators=params['n_estimators'],
            device=params['device'],
            learning_rate=params['learning_rate']
        )
        xgb_model.fit(X_train, y_train)
        logger.debug('Model training completed')
        return xgb_model
    except KeyError as e:
        logger.error('Missing model param: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, output_dir: str):
    """Save the trained model to a pickle file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        logger.debug('Model saved to %s', output_dir)
    except Exception as e:
        logger.error('Error saving model: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        X_train, y_train = load_data('./data/features/train_bow.csv')
        xgb_model = train_model(X_train, y_train, params)
        save_model(xgb_model, "1. data-versioning")
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()