# Sentiment Analysis - E-Commerce

## Project Overview
This project is a sentiment analysis application developed to classify tweets from an e-commerce context as positive or negative. The analysis uses a deep learning approach with a bidirectional LSTM model trained on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).

## Problem Statement
Build a sentiment analysis model to classify the polarity (positive or negative) of tweets using a deep learning technique.

## Dataset
- **Source**: [Sentiment140 dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Description**: The dataset contains 1,600,000 tweets with binary sentiment labels (0 for negative and 1 for positive).

## Project Structure
1. **Data Collection and Exploration**: Load and inspect the dataset, including initial statistics and visualizations.
2. **Data Preprocessing**:
   - Cleaning text (removal of @mentions, RTs, URLs, hashtags, non-alphabetic characters, etc.)
   - Tokenization and padding of tweets.
3. **Model Architecture**:
   - A bidirectional LSTM network with a pre-trained GloVe embedding layer.
   - Hyperparameters like `max_words`, `max_len`, `embedding_dim`, `batch_size`, and `epochs`.
4. **Model Training**:
   - Splitting data into training and validation sets.
   - Training and validating the model with accuracy and loss tracking.
5. **Evaluation**:
   - Performance metrics on test data (accuracy, precision, recall, F1-score).
   - Confusion matrix for detailed error analysis.

## Requirements
- Python
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow`, `keras`, `scikit-learn`

## Key Findings
- The model achieved an accuracy of 82% on the test set.
- It demonstrated balanced performance across both positive and negative sentiment classes.
- Confusion matrix analysis showed a low false positive and false negative rate, indicating good generalization capability.

## Usage
1. Clone the repository.
2. Install the required libraries.
3. Run the Jupyter notebook to load data, preprocess it, train the model, and evaluate results.

## Future Steps
1. **Hyperparameter Tuning**: Experiment with different hyperparameters (e.g., batch size, learning rate, LSTM layers) to optimize model performance.
2. **Alternative Architectures**: Explore advanced architectures, such as transformer-based models (e.g., BERT or RoBERTa), for improved sentiment classification accuracy.
3. **Data Augmentation**: Use data augmentation techniques like synonym replacement or back-translation to diversify training data and reduce overfitting.
4. **Multi-Class Sentiment Analysis**: Extend the model to classify more nuanced sentiments (e.g., neutral, very positive, very negative).
5. **Deployment**: Package the model into a deployable format using tools like Flask or FastAPI and create a web or mobile app interface for real-time sentiment analysis on new tweets.

## Sample Predictions
The model accurately predicts sentiment for various sample tweets, demonstrating its practical utility in classifying e-commerce-related sentiments.
