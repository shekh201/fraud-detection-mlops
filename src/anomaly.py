from sklearn.ensemble import IsolationForest


def create_anomaly_model(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1
):
    
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    return model