# 🎯 Classification: Logistic Regression from Scratch

This folder contains implementations of **Logistic Regression**, a fundamental classification algorithm. These models are built using **NumPy** to demonstrate the mechanics of binary classification, probability mapping, and overfitting prevention.

## 🧠 Key Mathematical Concepts

### 1. The Sigmoid Function (Activation)
Unlike linear regression, logistic regression uses the **Sigmoid Function** to map any real-valued number into a probability range between 0 and 1:
$$g(z) = \frac{1}{1 + e^{-z}}$$
A prediction is made as $\hat{y} = 1$ when $g(z) \geq 0.5$ (which occurs when $z \geq 0$).

### 2. Logistic Loss (Binary Cross-Entropy)
We use the **Log Loss** function to penalize incorrect predictions. This ensures that the cost increases significantly when the model is confident but wrong:
$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)}))]$$

### 3. Regularization ($\lambda$)
To prevent **Overfitting (High Variance)**, we apply L2 Regularization. This adds a penalty term to the cost function, discouraging the weights $w$ from becoming too large and making the decision boundary "wiggly".

---

## 📁 File Descriptions

* **`logistic_regression.py`**: The standard implementation for binary classification. It includes:
    * Gradient Descent optimization.
    * Decision boundary visualization for 2D features.
* **`logistic_regression_regularized.py`**: An advanced version that includes the **Regularization Parameter ($\lambda$)** to improve the model's ability to generalize to new data.

---

## 📊 Visualizing the Decision Boundary


The decision boundary is the line where the probability $P(y=1|x) = 0.5$. In these scripts, we visualize this boundary to see how effectively the model separates the different classes in the training data.

## 💻 Requirements
* Python 3.x
* NumPy
* Matplotlib
