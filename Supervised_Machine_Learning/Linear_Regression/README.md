# 🚀 Machine Learning: Linear Regression from Scratch

This repository contains implementations of **Linear Regression** built from the ground up using **NumPy** and **Matplotlib**. It demonstrates the transition from simple single-variable models to advanced multivariate models with feature scaling.

## 📂 Project Structure
* `linear_regression_model.py`: Simple Linear Regression (one feature).
* `Linear_regression_multivariable_model.py`: Multiple Linear Regression with Z-Score Normalization.
* `README.md`: Project documentation and mathematical overview.

---

## 🧠 Mathematical Foundations

### 1. Prediction (Hypothesis) Function
For multiple variables, the model predicts the output using the dot product of the input vector $\vec{x}$ and weight vector $\vec{w}$:
$$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$


### 2. Cost Function (Mean Squared Error)
To measure accuracy, we calculate the average squared difference between predictions and actual targets:
$$J(\vec{w},b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2$$


### 3. Gradient Descent Optimization
We update parameters simultaneously to minimize the cost function:
* $w_j = w_j - \alpha \frac{\partial J(\vec{w},b)}{\partial w_j}$
* $b = b - \alpha \frac{\partial J(\vec{w},b)}{\partial b}$

---

## 🛠️ Key Features Implemented

* **Multiple Features:** The model handles multiple inputs (e.g., Size, Bedrooms, Age) using vectorized NumPy operations.
* **Z-Score Normalization:** A critical preprocessing step that scales features to have a mean of 0 and a standard deviation of 1. This prevents numerical overflow and ensures smooth convergence.
* **Convergence Monitoring:** The script records the cost at every 100 iterations to generate a Learning Curve, allowing us to verify that the error is decreasing over time.
* **Data Visualization:** Individual scatter plots for each feature vs. the target, including the model's line of best fit to verify accuracy.

---

## 📊 Performance Visualization


When the model is trained correctly, the **Cost vs. Iterations** graph should show a steep decline that levels off (converges) as it finds the optimal parameters. If the cost increases, it indicates the learning rate $\alpha$ is too high.

## 💻 Requirements
* Python 3.x
* NumPy
* Matplotlib
