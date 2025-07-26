from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

def train_model(X_train, y_train):
    model = MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight='balanced'))
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, mlb=None):
    y_pred = model.predict(X_test)
    y_sent_true, y_topics_true = y_test[:, 0], y_test[:, 1:]
    y_sent_pred, y_topics_pred = y_pred[:, 0], y_pred[:, 1:]

    report = {}
    report['sentiment_accuracy'] = accuracy_score(y_sent_true, y_sent_pred)
    report['topic_precision'] = precision_score(y_topics_true, y_topics_pred, average='micro')
    report['topic_recall'] = recall_score(y_topics_true, y_topics_pred, average='micro')
    report['topic_f1'] = f1_score(y_topics_true, y_topics_pred, average='micro')

    if mlb:
        print("\n Per-Topic Classification Report:")
        print(classification_report(y_topics_true, y_topics_pred, target_names=mlb.classes_))

    return report
