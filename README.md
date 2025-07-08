# Deep Learning-Based Spam Detection

This project presents a deep learning approach to spam message detection using a feedforward neural network (FFNN). The model is designed to classify messages as either "spam" or "ham" (non-spam) and is benchmarked against traditional machine learning algorithms such as Random Forest, SVM, and Naive Bayes.

## Key Highlights

- Built using a **custom FFNN** with embedding, pooling, dense, dropout, and output layers.
- Achieved an accuracy of **~93%** on the test dataset.
- Outperformed traditional ML models (Random Forest, SVM, Naive Bayes).
- Included early stopping and model checkpointing for robust training.
- Designed to be lightweight and deployable on smaller systems.
- Ready for **Hugging Face / Chrome extension integration** for real-time usage.


## Methodology

1. **Data Collection**:
   - Used diverse public datasets (e.g., Enron Email Dataset, UCI SMS Spam Collection).
   - Balanced distribution of spam and ham messages.
   - Included multilingual, sarcastic, phishing, and promotional spam.

2. **Preprocessing**:
   - Tokenization using TensorFlow tokenizer.
   - TF-IDF vectorization.
   - Label encoding and out-of-vocabulary handling.

3. **Model Architecture**:
   - **Embedding Layer** (50 tokens, 16-dim)
   - **Global Average Pooling**
   - **Dense Layer (32 neurons)**
   - **Dropout Layer (rate = 0.3)**
   - **Output Layer (Sigmoid, binary classification)**

4. **Training Strategy**:
   - Trained for 30 epochs with early stopping (patience = 3).
   - Used `ModelCheckpoint` to retain best performing weights.
   - Evaluated using precision, recall, F1-score, accuracy, and confusion matrix.
