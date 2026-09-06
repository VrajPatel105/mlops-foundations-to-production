# 4. ml pipeline : model building

import os
import pandas as pd
import numpy as np

import xgboost as xgb
import pickle

OUTPUT_DIR = "1. data-versioning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_data = pd.read_csv('./data/features/train_bow.csv')

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=100,
    device="cpu"
)
xgb_model.fit(X_train, y_train)

pickle.dump(xgb_model, open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb"))