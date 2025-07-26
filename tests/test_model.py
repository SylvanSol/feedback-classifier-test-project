import sys
import os

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from src import preprocess, model

def test_model_pipeline():
    df = preprocess.load_and_clean_data("data/dummy_feedback.csv")
    X, _ = preprocess.prepare_features(df)
    y_sent, y_topics, _ = preprocess.prepare_labels(df)
    y_combined = np.hstack([y_sent, y_topics])

    X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.2)
    clf = model.train_model(X_train, y_train)
    results = model.evaluate_model(clf, X_test, y_test)

    assert results["sentiment_accuracy"] >= 0.0  # simple smoke test
