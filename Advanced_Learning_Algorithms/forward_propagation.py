import numpy as np

#Signmoid Function
def g(z):

    z = np.clip(z, -500, 500)
    g = 1 / (1 + np.exp(-z))    
    return g


#Dense Function ,Calculates layer by layer.
def dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    
    for j in range(units):
        w = W[:,j]
        z = np.dot(w,a_in) + b[j]
        a_out[j] = g(z) # applying sigmoid function
        
    return a_out
    
#Sequential Function ,Calculates output of whole neural network by calling dense function for each layer.
def sequential(x):
    a1 = dense(x,W1,b1)
    a2 = dense(a1,W2,b2)

    return a2



#Layer 1 parameters
W1 = np.array([[-8.93,  0.29, 12.9 ], 
               [-0.1,  -8.31, 10.81]])
b1 = np.array([-9.47, +8.93, -4.1])

#Layer 2 parameters
W2 = np.array([[-31.18], 
               [-27.59], 
               [-32.56]])
b2 = np.array([15.41])

# Testing the model with a sample input
x_test = np.array([200, 13.9])
prediction = sequential(x_test)
print("Network Prediction:", prediction)