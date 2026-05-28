# 🧠 Course 2: Advanced Learning Algorithms

This folder contains implementations from **Course 2** of the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) by Andrew Ng on Coursera.

---

## 📂 Files

| File | Description |
|---|---|
| `forward_propagation.py` | Manual forward pass implemented in NumPy — sigmoid activation, `dense()` layer, and `sequential()` network |
| `train_neural_network_tensorflow.py` | Binary classifier built and trained using TensorFlow/Keras (`Sequential`, `Dense`, `BinaryCrossentropy`) |
| `train_neural_network_softmax.py` | Multiclass classifier built and trained using TensorFlow/Keras with Softmax and linear activation for numerical stability |

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
- **`SparseCategoricalCrossentropy`** — loss function for multiclass classification problems where target labels are integers
- **`model.compile()`** — sets the optimizer and loss
- **`model.fit()`** — runs gradient descent over epochs to learn weights

### Numerical Stability (Softmax / from_logits)
- For multiclass output, using a `linear` activation in the final output layer and setting `from_logits=True` in the loss function reduces round-off errors by allowing TensorFlow to optimize the loss calculation.
- During inference, `tf.nn.softmax()` is applied to the output logits to obtain probabilities.

---

## 🔲 Upcoming
- Decision Trees & Ensembles
- Bias / Variance diagnostics
- Regularization in neural networks
