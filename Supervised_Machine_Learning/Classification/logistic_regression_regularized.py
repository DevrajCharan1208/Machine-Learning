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
def cost_function_reg(x,y,w,b,lambda_):
    m,n = x.shape
    
    cost = 0 
    reg_cost = 0
    
    for i in range(m):
        z = np.dot(x[i],w) + b
        f_wb = sigmoid(z)
        
        cost += y[i]*np.log(f_wb) + (1-y[i])*np.log(1-f_wb)
    for j in range (n):
        reg_cost += w[j]**2
        
    reg_cost = (lambda_*reg_cost)/(2*m)
    cost = -cost/m
    
    total_cost = cost + reg_cost
    return total_cost

#calculating gradient for w , b 
def gradient_func_reg(x,y,w,b,lambda_):
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
    
    
    dw = (dw/m) 
    db = db/m
    
    for j in range(n):
        dw[j] += w[j]*(lambda_/m)
        
    return dw, db

#applying gradiend descent 
def gradient_descent_reg(x,y,w_in,b_in,cost_function,gradient_function,alpha,iterations,lambda_):
    
    j_hist = [] #to recort history of cost function
    w = w_in.copy()
    b = b_in
    
    for i in range(iterations):
        dw, db = gradient_function(x,y,w,b,lambda_)
        
        w = w - alpha*dw
        b = b - alpha*db
        
        J = cost_function(x,y,w,b,lambda_)
        j_hist.append(J)
        
        if (i%5000 == 0):
            print(f" for iteration no. {i}, cost function (J) is {J}")
            print(f"value of w is {w}, b is {b}")
            
    return w, b, j_hist


#=========================DATA INTIALIZATION=======================

def map_feature(x1, x2, degree=6):

    out = []
    
    if np.isscalar(x1):
        x1 = np.array([x1])
        x2 = np.array([x2])
        
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((x1 ** (i - j)) * (x2 ** j))
            
  
    return np.column_stack(out)





# X_train: 28 examples, 2 features each
X_train = np.array([
    [0.051267, 0.69956],  [-0.092742, 0.68494], [-0.21371, 0.69225],  [-0.375, 0.50219], 
    [-0.51325, 0.46564],  [-0.52477, 0.2098],   [-0.39804, 0.034357], [-0.30588, -0.19225],
    [0.016705, -0.40424], [0.13191, -0.51389],  [0.38537, -0.56506],  [0.52938, -0.5212],
    [0.63882, -0.24342],  [0.73675, -0.18494],  [0.54666, 0.48757],   [0.322, 0.5826],    
    [0.16647, 0.53874],   [-0.046659, 0.81652], [-0.17339, 0.69956],  [-0.47869, 0.63377],
    [-0.60541, 0.59722],  [-0.62846, 0.33406],  [-0.59389, 0.005117], [-0.42108, -0.27266],
    [-0.11578, -0.39693], [0.20104, -0.60161],  [0.46601, -0.53582],  [0.67339, -0.53582]
])


y_train = np.array([
    1, 1, 1, 1, 
    1, 1, 1, 1, 
    1, 1, 1, 1, 
    1, 1, 0, 0, 
    0, 0, 0, 0, 
    0, 0, 0, 0, 
    0, 0, 0, 0
])
X_new = map_feature(X_train[:, 0], X_train[:, 1], degree=8)

w_initial = np.ones(X_new.shape[1])
b_initial = 0.1
alpha = 0.5
iters = 100000
lambda_ = 0.01


w, b, j_hist = gradient_descent_reg(X_new,y_train,w_initial,b_initial,cost_function_reg,gradient_func_reg,alpha,iters,lambda_)



print (f"Final value for : w - {w}, b - {b}")




#=============================PREDICTING TEST==========================
#correcet output should be 0.
#outputs recorded (L -> lambda) - [L,output] => [0.1, 0][0.01, 0](just right fitting), [0,0](overfitting), [1,0](underfitting)
x1_test = 0.5
x2_test = 0.8


x_test_mapped = map_feature(x1_test, x2_test, degree=8)


pred_value = prediction(x_test_mapped, w, b)


print(f"Prediction for (0.5, 0.8): {pred_value[0]}")

#=====================ACCURACY======================================
#Accuracy recorded (L -> lambda) - [L,Accuracy] => [0.1, 67.86%][0.01,78.75%](just right fitting), [0,92.86%](overfitting), [1,60.71%](underfitting)
p = prediction(X_new, w, b)


accuracy = np.mean(p == y_train) * 100

print(f"Train Accuracy: {accuracy:.2f}%")
#===========================PLOTTING=================================

plt.plot(j_hist)
plt.title("Cost vs. Iteration")
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', s=100, label='Data')

# Define the Grid
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))

# Calculate Prediction for every point on the grid
for i in range(len(u)):
    for j in range(len(v)):
        # KEY CHANGE: We use the same map_feature function inside the loop!
        # This ensures the grid points have the same 27 features as training data.
        feature_vector = map_feature(u[i], v[j], degree=8)
        
        z[i,j] = np.dot(feature_vector, w) + b

# Draw the Contour
ax.contour(u, v, z.T, levels=[0], colors='green', linewidths=2)
ax.set_title("Decision Boundary (Overfitting with Degree 8)")
plt.show()