from sklearn.model_selection import train_test_split
import os, pickle, sys
import pandas as pd
import numpy as np

# Add the necessary imports for the starter code.
dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dir, 'ml'))
from model import train_model, compute_model_metrics, inference
from data import process_data

# Add code to load in the data.
data = pd.read_csv('data/census.csv')
for col in data.columns:
    col_new = col.strip()
    if col != col_new:
        data[col_new] = data[col]
        data.drop([col], axis = 1, inplace = True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size = .2)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb = lb
)

# Train and save a model.
model = train_model(X_train, y_train)
pickle.dump(model  , open('./model/model.pkl'  , 'wb'))
pickle.dump(encoder, open('./model/encoder.pkl', 'wb'))
pickle.dump(lb     , open('./model/lb.pkl'     , 'wb'))

preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

def slice_output(df, feature_slice):
    f = open('./model/slice_output.txt', 'w')
    for cls in df[feature_slice].unique():
        idx = df[feature_slice] == cls
        precision, recall, fbeta = compute_model_metrics(y_test[idx], preds[idx])
        f.write(f"Class: {feature_slice} = {cls} \n")
        f.write(f"Precision: {precision:.4f} \n")
        f.write(f"Recall: {recall:.4f} \n")
        f.write(f"Fbeta: {fbeta:.4f} \n")
        f.write(f"\n")
    f.close()

slice_output(test, 'education')
