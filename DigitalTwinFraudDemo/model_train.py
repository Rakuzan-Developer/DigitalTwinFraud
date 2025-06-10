# model_train.py
from sklearn.ensemble import IsolationForest

def fit_isolation_forest(df_trans):
    X = df_trans[['tutar', 'saat']]
    clf = IsolationForest(contamination=0.02, random_state=42)
    clf.fit(X)
    preds = clf.predict(X)
    anomaly_flag = (preds == -1).astype(int)
    df_trans = df_trans.copy()
    df_trans['ml_anomaly_flag'] = anomaly_flag
    return df_trans
