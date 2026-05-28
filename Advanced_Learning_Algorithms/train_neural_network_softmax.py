import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import numpy as np


X = np.random.rand(100,20)
Y = np.random.randint(0,10,size=(100,))

#Define Model

x = np.random.rand(1,20)

model = Sequential([
    Dense(units=25,activation='relu'),
    Dense(units=15,activation='relu'),
    Dense(units=10,activation='linear') #to reduce round of errors
])

#loss
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))

# Fit the model
model.fit(X,Y,epochs=100)

#predict
logits = model(x)
f_x = tf.nn.softmax(logits)

print("\n====Output====\n")
print(f_x)