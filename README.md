# Loan Approval Prediction using Machine Learning

A supervised machine-learning project that predicts whether a loan application is likely to be approved based on applicant data.

Instead of relying on a single model, the project compares multiple classification algorithms and evaluates their performance.

## Models Compared

- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Classifier

## Data Pipeline

```text
Loan Application Data
        ↓
Missing-Value Handling
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Train/Test Split
        ↓
Multiple Classifiers
        ↓
Performance Comparison
```

## Results

The current notebook reports the following result for the best-performing model:

| Model | Accuracy | Precision |
|---|---:|---:|
| Decision Tree | **92%** | **83.58%** |

These are the results reported by the current notebook and may vary if the preprocessing or random state is changed.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Jupyter Notebook

## Getting Started

```bash
git clone https://github.com/muskanmundra18-lab/loan-approval.git
cd loan-approval
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

Open:

```text
minor_project_1.ipynb
```

## Key Learning Outcomes

- Handling missing numerical and categorical values
- Encoding categorical features
- Feature scaling
- Comparing classification algorithms
- Evaluating models with multiple metrics
- Understanding model selection rather than relying on a single algorithm

## Future Improvements

- Add ROC-AUC and confusion matrices
- Use cross-validation for more reliable comparison
- Add explainability with SHAP
- Build a prediction API
- Add fairness and bias analysis before real-world use

## Author

**Muskan Mundra**
