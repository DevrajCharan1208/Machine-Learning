import numpy as np
import matplotlib.pyplot as plt

#training data - features
x_train = np.array([
    [2104, 5, 45],
    [1416, 3, 40],
    [1534, 3, 30],
    [852,  2, 36],
    [1200, 2, 20],  
    [2500, 4, 10]   
])

#training data - targets
y_train = np.array([460, 232, 315, 178, 240, 550])

#function to predict taret value for new inputs
def prediction(x,w,b):
    f_wb = np.dot(x,w) + b
    return f_wb

#function to calculate cost of the model
def cost(x,y,w,b):
    m = x.shape[0]
    cost = 0 
    
    for i in range(m):
        f_wb = np.dot(x[i],w) + b
        cost += (f_wb - y[i])**2
    cost = cost/(2*m)
    
    return cost

#function to calculate gradients 
def gradients(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    
    for i in range(m):
        error = (np.dot(x[i],w)+b) - y[i]
        for j in range(n):
            dj_dw[j] += error*x[i,j]
        dj_db += error
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    
    return dj_dw , dj_db

#function to apply gradient descent
def gradient_descent(x,y,w_in,b_in,cost,gradient,alpha,iterations):
    j_hist = []
    w = w_in.copy()
    b = b_in 
    
    for i in range(iterations):
        dj_dw , dj_db = gradient(x,y,w,b)
        
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
        if (i % 100 == 0):
            
            j_hist.append(cost(x,y,w,b))
            print(f"iteration - {i}    ,    Cost - {j_hist[-1]}")

    return w , b , j_hist

#funciton to normalize values of features
def z_norm(x):
    
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    
    x_norm = (x-mu) / sigma
    
    return (x_norm,mu,sigma)


#intial data
w_in = [0,0,0]
b_in = 0
x_norm, x_mu, x_sigma = z_norm(x_train)

iterations = 10000
alpha = 0.001
# w, b, j_hist = gradient_descent(x_train,y_train,w_in,b_in,cost,gradients,alpha,iterations) #without normalization - very small alpha value required
w, b, j_hist = gradient_descent(x_norm,y_train,w_in,b_in,cost,gradients,alpha,iterations)

print(w,b)


y_pred = prediction(x_norm,w,b)


#plots to check sanity of model
# Feature 1 (x1) vs y
plt.scatter(x_norm[:, 0], y_train) # Actual data
plt.plot(x_norm[:, 0], y_pred, color='orange') # Model prediction line
plt.xlabel("x1 (Normalized)")
plt.ylabel("y")
plt.title("Feature x1 Fit")
plt.show()

# Feature 2 (x2) vs y
plt.scatter(x_norm[:, 1], y_train)
plt.plot(x_norm[:, 1], y_pred, color='orange')
plt.xlabel("x2 (Normalized)")
plt.ylabel("y")
plt.title("Feature x2 Fit")
plt.show()

# Feature 3 (x3) vs y
plt.scatter(x_norm[:, 2], y_train)
plt.plot(x_norm[:, 2], y_pred, color='orange')
plt.xlabel("x3 (Normalized)")
plt.ylabel("y")
plt.title("Feature x3 Fit")
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(j_hist)
plt.title("Cost Function vs. Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
