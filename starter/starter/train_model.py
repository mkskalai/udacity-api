# Script to train machine learning model.
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump, load

# Add the necessary imports for the starter code.
from ml.data import load_data, process_data
from ml.model import train_model, compute_model_metrics, inference
from evaluate_slices import evaluate_cat_slices, evaluate_all_cats


dir_path = os.path.dirname(os.path.realpath(__file__))

# Add code to load in the data.
data = load_data()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
train.to_csv(os.path.join(dir_path, '../data/data_train.csv'), index=False)
test.to_csv(os.path.join(dir_path, '../data/data_test.csv'), index=False)

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
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
dump(model, os.path.join(dir_path, '../model/random_forest.joblib'))
dump(encoder, os.path.join(dir_path, '../model/encoder.joblib'))
dump(lb, os.path.join(dir_path, '../model/lb.joblib'))

# run evaluation
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Overall performance:\nprecision: {precision:<9.3f} recall: {recall:<9.3f} fbeta: {fbeta:<9.3f}")
slice_eval = evaluate_all_cats(test, cat_features, "salary", model, encoder, lb)
with open(os.path.join(dir_path, '../model/slice_output.txt'), "w") as f:
    f.writelines(slice_eval)
