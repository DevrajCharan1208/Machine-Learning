import pandas as pd
import matplotlib.pyplot as plt

#data for input and output label
df = pd.read_csv('data.csv')
x = df.iloc[:,0]
y = df.iloc[:,1]


#fucntion to calculate cost function J = 1/2m.sum((f(x)-y)^2)
def cost_func(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w*x[i] + b
        cost += (f_wb - y[i])**2
    total_cost = cost/(2*m)
    return total_cost


#function to calculate gradient values dJ/dw, dJ/db for gradient descent algo
def gradient_func(x,y,w,b):
    m = x.shape[0]
    grad_w = 0 #dJ/dw
    grad_b = 0 #dJ/db
    
    for i in range(m):
        f_wb = w*x[i] + b
        grad_w_temp = (f_wb - y[i])*x[i]
        grad_b_temp = (f_wb - y[i])
        grad_w += grad_w_temp
        grad_b += grad_b_temp
    grad_w = grad_w/m
    grad_b = grad_b/m
    return grad_w,grad_b


#function to calculate gradient descent
def gradient_descent(x,y,w,b,alpha,iterations,cost_func,grad_func):
    J_hist = []
    P_hist = []
    for i in range(iterations):
        #calculate gradients
        grad_w , grad_b = grad_func(x,y,w,b)
        
        #update values of w & b 
        w = w - alpha*grad_w
        b = b - alpha*grad_b
        
        J = cost_func(x,y,w,b)
        J_hist.append(J)
        P_hist.append([w,b])
        if (i % 500 == 0):
            print(f"after number of {i} iterations, Cost is {J:.4f}\n w,b are {w:.3f},{b:.3f} ")    
    return w , b, J_hist, P_hist


#initial parameters
w = 0
b = 0
iterations = 10000
alpha = 0.0001

#running gradient descent
w,b,j_hist,p_hist = gradient_descent(x,y,w,b,alpha,iterations,cost_func,gradient_func)

print(f"(w,b found by gradient descent are: {w:.3f}, {b:.3f})")



plt.scatter(x,y)
plt.plot(x,w*x+b,color="red")
plt.title("Linear Regression Line")
plt.show()









