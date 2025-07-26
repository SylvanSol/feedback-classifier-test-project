import sys
import os

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import preprocess, model
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    print("Loading and cleaning data...")
    df = preprocess.load_and_clean_data("data/dummy_feedback.csv")

    print("Extracting features...")
    X, vectorizer = preprocess.prepare_features(df)

    print("Preparing labels...")
    y_sent, y_topics, mlb = preprocess.prepare_labels(df)
    y_combined = np.hstack([y_sent, y_topics])

    print("Splitting train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.2, random_state=42)

    print("Training model...")
    clf = model.train_model(X_train, y_train)

    print("Evaluating model...")
    results = model.evaluate_model(clf, X_test, y_test)

    print("\n--- Evaluation Results ---")
    print(f"Sentiment Accuracy: {results['sentiment_accuracy']:.2f}")
    print(f"Topic Precision: {results['topic_precision']:.2f}")
    print(f"Topic Recall: {results['topic_recall']:.2f}")
    print(f"Topic F1 Score: {results['topic_f1']:.2f}")

if __name__ == "__main__":
    main()
