# 5. ml pipeline : model evaluation final stage for ml pipeline

import os
import json
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

OUTPUT_DIR = "1. data-versioning"

xgb = pickle.load(open(os.path.join(OUTPUT_DIR, "model.pkl"), "rb"))

test_data = pd.read_csv("./data/features/test_bow.csv")

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "roc_auc": auc
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_dict, f, indent=4)