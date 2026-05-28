# 🤖 Machine Learning Specialization — Andrew Ng (Coursera)

This repository contains my personal notes, implementations, and experiments from the **Machine Learning Specialization** by Andrew Ng on Coursera. Core algorithms are built **from scratch using NumPy** to develop a deep understanding of the underlying mathematics, with TensorFlow used where the course introduces it.

---

## 📚 Course Progress

### ✅ Course 1: Supervised Machine Learning — Regression and Classification

| Topic | Folder | Status |
|---|---|---|
| Linear Regression (Single & Multi-variable) | `Supervised_Machine_Learning/Linear_Regression/` | ✅ Done |
| Logistic Regression & Regularization | `Supervised_Machine_Learning/Classification/` | ✅ Done |

### 🟡 Course 2: Advanced Learning Algorithms

| Topic | File | Status |
|---|---|---|
| Forward Propagation (NumPy from scratch) | `Advanced_Learning_Algorithms/forward_propagation.py` | ✅ Done |
| Training a Neural Network (TensorFlow) | `Advanced_Learning_Algorithms/train_neural_network_tensorflow.py` | ✅ Done |
| Multiclass Classification (Softmax) | `Advanced_Learning_Algorithms/train_neural_network_softmax.py` | ✅ Done |
| Decision Trees | `Advanced_Learning_Algorithms/` | 🔲 Upcoming |

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
└── Advanced_Learning_Algorithms/
    ├── forward_propagation.py                  # Manual forward pass using NumPy (sigmoid, dense, sequential)
    ├── train_neural_network_tensorflow.py      # Binary classifier trained with TF/Keras (BinaryCrossentropy)
    ├── train_neural_network_softmax.py         # Multiclass classifier trained with TF/Keras (SparseCategoricalCrossentropy)
    └── README.md                               # Concepts & implementation notes for Course 2
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Libraries:** NumPy, Matplotlib, TensorFlow / Keras
- **No Scikit-Learn** — core algorithms implemented from scratch
- **TensorFlow** — used for higher-level neural network training (as introduced in Course 2)

---

## 🧠 Key Concepts Covered So Far

**Course 1 — Supervised Machine Learning**
- **Gradient Descent** — parameter optimization with learning rate tuning
- **Z-Score Normalization** — feature scaling for faster convergence
- **Sigmoid Function** — mapping linear output to probabilities
- **Log Loss (Binary Cross-Entropy)** — cost function for classification
- **L2 Regularization** — preventing overfitting via weight penalty
- **Polynomial Feature Mapping** — enabling non-linear decision boundaries
- **Bias-Variance Tradeoff** — understanding underfitting vs. overfitting

**Course 2 — Advanced Learning Algorithms**
- **Forward Propagation** — manual layer-by-layer computation using NumPy (dense + sequential)
- **Neural Network Architecture** — stacking Dense layers with sigmoid, relu, and linear activations
- **TensorFlow / Keras** — building and compiling models with `Sequential`, `Dense`, `BinaryCrossentropy`, and `SparseCategoricalCrossentropy`
- **Model Training** — using `model.compile()` and `model.fit()` for end-to-end learning
- **Numerical Stability (Softmax)** — using `linear` activation in the final layer with `from_logits=True` to reduce round-off errors
