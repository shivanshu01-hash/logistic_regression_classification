# Logistic Regression Classification

This project demonstrates the implementation of **Logistic Regression** for a binary classification problem using Python and scikit-learn.

## Project Overview
- Preprocessed dataset using StandardScaler
- Trained a Logistic Regression model
- Evaluated performance with:
  - Confusion Matrix
  - Accuracy Score
  - Classification Report
- Visualized:
  - Confusion Matrix Heatmap
  - Decision Boundary for Training and Test sets
- Analyzed Bias vs Variance for model insights

## Results
- Accuracy: **92.5%**
- Confusion Matrix: [[57, 1], [5, 17]]
- Slight underfitting observed, but overall strong performance.

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## How to Run
1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python logistic_regression_classification.py
   ```

## Visualizations
- Confusion Matrix Heatmap
- Decision Boundary plots (Train & Test)

## Insights
- Logistic Regression performed well with 92.5% accuracy
- Slight underfitting, indicating potential improvement
- Visualization helps interpret the model beyond metrics

## Author
Developed by Shivanshu Sahu
