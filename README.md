# 🤖 Machine Learning Specialization — Andrew Ng (Coursera)

This repository contains my personal notes, implementations, and experiments from the **Machine Learning Specialization** by Andrew Ng on Coursera. All models are built **from scratch using NumPy** — no Scikit-Learn — to develop a deep understanding of the underlying mathematics.

---

## 📚 Course Progress

### ✅ Course 1: Supervised Machine Learning — Regression and Classification

| Topic | Folder | Status |
|---|---|---|
| Linear Regression (Single & Multi-variable) | `Supervised_Machine_Learning/Linear_Regression/` | ✅ Done |
| Logistic Regression & Regularization | `Supervised_Machine_Learning/Classification/` | ✅ Done |

### 🔲 Course 2: Advanced Learning Algorithms

| Topic | Folder | Status |
|---|---|---|
| Neural Networks | `Advanced_learning_algorithms/` | 🔲 Upcoming |
| Decision Trees | `Advanced_learning_algorithms/` | 🔲 Upcoming |

### 🔲 Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning
> Upcoming

---

## 📂 Repository Structure

```
Machine-Learning/
│
├── Supervised_Machine_Learning/
│   ├── Linear_Regression/
│   │   ├── single_variable.py          # Simple Linear Regression (1 feature)
│   │   ├── multi_variable.py           # Multiple Linear Regression + Z-Score Normalization
│   │   ├── data.csv                    # Training dataset
│   │   └── README.md                   # Math, concepts & implementation details
│   │
│   └── Classification/
│       ├── logistic_regression.py              # Binary Classification (linearly separable)
│       ├── logistic_regression_regularized.py  # Polynomial features + L2 Regularization
│       ├── decision_boundary_optimal.png       # λ = 0.01 — Best fit
│       ├── decision_boundary_overfit.png       # λ = 0    — Overfitting
│       ├── decision_boundary_underfit.png      # λ = 1    — Underfitting
│       └── README.md                           # Bias-Variance tradeoff experiment
│
└── Advanced_learning_algorithms/       # 🔲 Coming soon
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Libraries:** NumPy, Matplotlib
- **No Scikit-Learn** — everything implemented from scratch

---

## 🧠 Key Concepts Covered So Far

- **Gradient Descent** — parameter optimization with learning rate tuning
- **Z-Score Normalization** — feature scaling for faster convergence
- **Sigmoid Function** — mapping linear output to probabilities
- **Log Loss (Binary Cross-Entropy)** — cost function for classification
- **L2 Regularization** — preventing overfitting via weight penalty
- **Polynomial Feature Mapping** — enabling non-linear decision boundaries
- **Bias-Variance Tradeoff** — understanding underfitting vs. overfitting
