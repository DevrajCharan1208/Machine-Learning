# Linear Regression from Scratch

This project implements **Simple Linear Regression** using the **Gradient Descent** optimization algorithm without using high-level ML libraries like Scikit-Learn.

## 🧠 The Mathematical Model

### 1. Prediction Function
We use a linear model to predict the output:
$$f_{w,b}(x) = wx + b$$

### 2. Cost Function (Mean Squared Error)
To measure how well our model fits the data, we calculate the average squared error:
$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

### 3. Gradient Descent Updates
To minimize the cost, we calculate the partial derivatives and update parameters $w$ and $b$ simultaneously:
$$w = w - \alpha \frac{\partial J}{\partial w}$$
$$b = b - \alpha \frac{\partial J}{\partial b}$$



## 📊 Usage
The script `linear_regression.py` performs the following:
1.  Loads data from `data.csv`.
2.  Runs Gradient Descent for $10,000$ iterations.
3.  Prints the final weight ($w$) and bias ($b$).
4.  Plots the training data and the resulting regression line.

### Requirements
* Python 3.x
* Pandas
* Matplotlib
