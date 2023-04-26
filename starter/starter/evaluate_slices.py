from ml.data import process_data
from ml.model import compute_model_metrics, inference

def evaluate_cat_slices(df, cat_features, label, cat, model, encoder, lb):
    values = df[cat].unique()
    results = []
    for v in values:
        df_tmp = df[df[cat] == v]
        X, y, _, _ = process_data(
            df_tmp, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
        )
        pred = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, pred)
        result_tmp = (
            f"Category: {cat: <20} value: {v: <30} precision: {precision:<9.3f} recall: {recall:<9.3f} fbeta: {fbeta:<9.3f}"
        )
        results.append(result_tmp)
    return "\n".join(results)

def evaluate_all_cats(df, cat_features, label, model, encoder, lb):
    results = []
    for cat in cat_features:
        results.append(evaluate_cat_slices(df, cat_features, label, cat, model, encoder, lb))
    return "\n".join(results)