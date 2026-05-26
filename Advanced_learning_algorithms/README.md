# 🧠 Course 2: Advanced Learning Algorithms

This folder contains implementations from **Course 2** of the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) by Andrew Ng on Coursera.

---

## 📂 Files

| File | Description |
|---|---|
| `Forward_Propagation.py` | Manual forward pass implemented in NumPy — sigmoid activation, `dense()` layer, and `sequential()` network |
| `Training_Neural_Network_Tensorflow.py` | Binary classifier built and trained using TensorFlow/Keras (`Sequential`, `Dense`, `BinaryCrossentropy`) |

---

## 🔑 Key Concepts

### Forward Propagation (from scratch)
- **Sigmoid activation** — maps any real value to (0, 1)
- **`dense()` function** — computes activations for a single layer: `a = g(Wa + b)`
- **`sequential()` function** — chains layers together to produce the final network output

### Neural Network Training (TensorFlow)
- **`Sequential` model** — stacks layers in order
- **`Dense` layer** — fully connected layer with configurable units and activation
- **`BinaryCrossentropy`** — loss function for binary classification problems
- **`model.compile()`** — sets the optimizer and loss
- **`model.fit()`** — runs gradient descent over epochs to learn weights

---

## 🔲 Upcoming
- Decision Trees & Ensembles
- Bias / Variance diagnostics
- Regularization in neural networks
