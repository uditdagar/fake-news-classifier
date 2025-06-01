# Fake News Detection using BiLSTM & TF-IDF

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

A deep learning model that classifies news articles as real or fake using Bidirectional LSTM (BiLSTM) and TF-IDF features, achieving 92% accuracy.

## Features

- Hybrid architecture combining BiLSTM and TF-IDF features
- Comprehensive text preprocessing pipeline
- Model evaluation with multiple metrics (Accuracy, F1-score, ROC-AUC)
- Detailed exploratory data analysis
- Modular code structure for easy extension

## Dataset

The model uses the [Kaggle Fake News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) containing:
- `True.csv`: 21,417 real news articles
- `Fake.csv`: 23,481 fake news articles

Each article contains:
- Title
- Text content
- Subject
- Date

## Model Architecture

![Model Architecture](assets/model_architecture.png) *(Optional: Add diagram if available)*

The hybrid model consists of:
1. **Text Processing Branch**:
   - Embedding Layer (128 dimensions)
   - Two Bidirectional LSTM layers (64 and 32 units)
   - Dropout layers for regularization

2. **TF-IDF Feature Branch**:
   - Dense layer (64 units)

3. **Combined**:
   - Concatenation of both branches
   - Final dense layers with sigmoid activation

## Performance Metrics

| Metric        | Score  |
|---------------|--------|
| Accuracy      | 99.1%  |
| F1-score      | 0.921  |
| ROC-AUC       | 0.976  |
| Precision     | 0.923  |
| Recall        | 0.919  |
