import tensorflow as tf
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import numpy as np

np.random.seed(42)

X = np.random.rand(200, 2) * 100  

Y = ((X[:, 0] > 50) & (X[:, 1] > 30)).astype(int)
Y = Y.reshape(-1, 1) 

print("X shape:", X.shape) 
print("Y shape:", Y.shape)

model = Sequential([
    Dense(units=25,activation='sigmoid'),
    Dense(units=15,activation='sigmoid'),
    Dense(units=1,activation='sigmoid')
])

from tensorflow.keras.losses import BinaryCrossentropy # type: ignore
model.compile(loss = BinaryCrossentropy())
model.fit(X,Y,epochs = 100)

X_new = np.array([[60.0, 45.0]])
prediction = model.predict(X_new)

print("Probability :", prediction)