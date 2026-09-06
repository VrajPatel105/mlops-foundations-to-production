# 4. ml pipeline : model building

import os
import pandas as pd

import xgboost as xgb
import pickle
import yaml

eval_metric = yaml.safe_load(open('params.yaml', 'r'))['model_building']['eval_metric']
n_estimators = yaml.safe_load(open('params.yaml', 'r'))['model_building']['n_estimators']
device = yaml.safe_load(open('params.yaml', 'r'))['model_building']['device']
learning_rate = yaml.safe_load(open('params.yaml', 'r'))['model_building']['learning_rate']

OUTPUT_DIR = "1. data-versioning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_data = pd.read_csv('./data/features/train_bow.csv')

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

xgb_model = xgb.XGBClassifier(
    eval_metric=eval_metric,
    n_estimators=n_estimators,
    device=device,
    learning_rate=learning_rate
)
xgb_model.fit(X_train, y_train)

pickle.dump(xgb_model, open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb"))