import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df['feedback'] = df['feedback'].str.lower().str.strip()
    df['topics'] = df['topics'].str.lower().str.strip()
    df['topics'] = df['topics'].apply(lambda x: [t.strip() for t in x.split(',')])

    topic_corrections = {
        "suport": "support",
        "sapport": "support",
        "blling": "billing",
        "billng": "billing",
        "delivry": "delivery",
        "dilivery": "delivery",
        "princing": "pricing",
        "prizing": "pricing",
        "pruduct quality": "product quality",
        "product qlty": "product quality",
        "uxx": "ux"
    }

    def correct_topics(topics):
        return [topic_corrections.get(t, t) for t in topics]

    df['topics'] = df['topics'].apply(correct_topics)
    return df
