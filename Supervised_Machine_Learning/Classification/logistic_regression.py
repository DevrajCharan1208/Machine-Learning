import numpy as np
import matplotlib.pyplot as plt



#calculating f(x), 
def prediction(x,w,b):
    m = x.shape[0]
    p = np.zeros(m)
    
    for i in range(m):
        
        z = np.dot(x[i],w) + b
        f_wb = sigmoid(z)

        if (f_wb >= 0.5):
            p[i] = 1
        else:
            p[i] = 0
            
    return p 


#calculating value of sigmoid function
def sigmoid(z):
    
    g = 1 / (1 + np.exp(-z))
    
    return g


#calculating cost function
def cost_function(x,y,w,b):
    m = x.shape[0]
    
    cost = 0 
    
    for i in range(m):
        z = np.dot(x[i],w) + b
        f_wb = sigmoid(z)
        
        cost += y[i]*np.log(f_wb) + (1-y[i])*np.log(1-f_wb)
        
    cost = -cost/m
    
    return cost

#calculating gradient for w , b 
def gradient_func(x,y,w,b):
    m,n = x.shape
    
    dw = np.zeros(w.shape)
    db = 0
    
    for i in range(m):
        z = np.dot(x[i],w) + b
        f_wb = sigmoid(z)
        error = f_wb - y[i]
        
        for j in range(n):
            dw[j] += error*x[i,j]
        db += error
    
    dw = dw/m
    db = db/m
    return dw, db

#applying gradiend descent 
def gradient_descent(x,y,w_in,b_in,cost_function,gradient_function,alpha,iterations):
    
    j_hist = [] #to recort history of cost function
    w = w_in.copy()
    b = b_in
    
    for i in range(iterations):
        dw, db = gradient_function(x,y,w,b)
        
        w = w - alpha*dw
        b = b - alpha*db
        
        J = cost_function(x,y,w,b)
        j_hist.append(J)
        
        if (i%5000 == 0):
            print(f" for iteration no. {i}, cost function (J) is {J}")
            print(f"value of w is {w}, b is {b}")
            
    return w, b, j_hist

X_train = np.array([
    [34.6, 78.0], 
    [30.2, 43.8], 
    [35.8, 72.9], 
    [60.1, 86.3], 
    [79.0, 75.3], 
    [45.0, 56.3], 
    [95.0, 40.0],
    [75.0, 80.0]
])

y_train = np.array([0, 0, 0, 1, 1, 0, 1, 1])


w_initial = np.ones(X_train.shape[1])
b_initial = 0.01
alpha = 0.002
iters = 100000

w, b, j_hist = gradient_descent(X_train,y_train,w_initial,b_initial,cost_function,gradient_func,alpha,iters)



print (f"Final value for : w - {w}, b - {b}")

plt.plot(j_hist)
plt.title("Cost vs. Iteration")
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', s=100, label='Data')
if len(w) == 2: 
    # Create two x-values  to draw the line
    x_values = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
    
    # Calculate corresponding y-values using the decision boundary formula
    y_values = -(w[0] * x_values + b) / w[1]
    
   
    ax2.plot(x_values, y_values, c='green', linewidth=3, label='Decision Boundary')
    
ax2.set_title("Decision Boundary")
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.legend()

plt.show()