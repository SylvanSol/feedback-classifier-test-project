# Feedback Classifier Test Project

## Overview
This project demonstrates an NLP-based text classification tool using Scikit-learn to automatically tag client feedback with sentiment (1–5 scale) and topic categories (multi-label).

## Features
- Dummy data generator (`generate_data.py`)
- Data cleaning and preprocessing (`preprocess.py`)
- TF-IDF feature extraction
- Multi-output logistic regression model with class weighting
- Model evaluation with per-topic classification report
- Jupyter notebooks for training pipeline and exploratory data analysis
- Prediction script for new feedback (`predict_feedback.py`)
- Example unit test (`tests/test_model.py`)

## File Structure
```
feedback-classifier-test-project/
├── data/
│   └── dummy_feedback.csv
├── notebooks/
│   ├── classifier_pipeline.ipynb
│   └── eda_and_insights.ipynb
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── main.py
│   └── predict_feedback.py
├── tests/
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/feedback-classifier-test-project.git
   cd feedback-classifier-test-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Training and Evaluation
Execute the full pipeline (data loading, training, evaluation):
```bash
python src/main.py
```

### Classify New Feedback
Run the prediction script and enter feedback when prompted:
```bash
python src/predict_feedback.py
```
The script will output a predicted sentiment score (1–5) and topic labels.

### Jupyter Notebooks
- **Training pipeline**: `notebooks/classifier_pipeline.ipynb`
- **Exploratory data analysis**: `notebooks/eda_and_insights.ipynb`

Launch a notebook server:
```bash
jupyter notebook notebooks/classifier_pipeline.ipynb
```

### Unit Tests
Run the example unit test:
```bash
pytest tests/test_model.py
```

## Model Improvements
- **Class weighting** for imbalanced labels (`class_weight='balanced'`)
- **Per-topic evaluation** using `classification_report`
- **Model persistence** with `joblib` (dump and load model)
- **Hyperparameter tuning** (e.g. `GridSearchCV`)
- **Transformer-based models** (e.g. BERT) for deeper language understanding
- **Web interface** with Streamlit or FastAPI for live prediction


## License
Unlicensed
