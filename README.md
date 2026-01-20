# Hybrid-Book-Recommendation-System
This repository contains an end-to-end recommendation engine designed to provide high-relevance book suggestions by overcoming traditional challenges like Data Sparsity and Popularity Bias.

### Consensus-Based Collaborative Filtering Engine

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

## 🛠️ System Architecture
This engine utilizes a **Hybrid Consensus Logic** to generate recommendations. Instead of relying on a single similarity metric, the system calculates and compares the outputs of **5 Nearest Neighbors (NN)** across three distinct mathematical models.

### 🧠 Similarity Models Implemented
The "Consensus Rank" is derived by aggregating scores from:
* **Jaccard Similarity:** Evaluates the binary intersection of user libraries.
* **Pearson Correlation:** Normalizes for user-rating bias and strictness.
* **Cosine Similarity:** Measures the vector orientation of item-rating profiles.



## 📊 Performance Metrics
* **Accuracy:** Variable accuracy due to limited number of books in the dataset.
* **Efficiency:** Measurable efficiency gains through optimized development and responsible AI adoption.
* **Validation:** Real-time **Hit Rate @ 5** pipeline used to verify suggestions against historical user interactions.

## 💻 Tech Stack
* **Backend:** Python, Flask
* **Data Science:** Scikit-Learn, SciPy, Pandas, NumPy
* **Frontend:** HTML

## ⚙️ Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sharma-rajat-02/book-recommender.git](https://github.com/sharma-rajat-02/book-recommender.git)
