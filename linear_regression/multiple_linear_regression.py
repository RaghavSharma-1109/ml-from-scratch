import numpy as np
np.random.seed(42)

X = np.random.randn(200,3)
# Corrrect Output
true_w = np.array([3,5,2])
true_b = 4
# Correct Prediction including noise
y = X@true_w + true_b + np.random.randn(200)*0.5

mean = np.mean(X,axis=0)
std = np.std(X,axis=0)

X = (X-mean)/std
# Model Predcition Process starts here
w = np.zeros(3)
b=0
alpha = 0.01
iterations =1000
n = len(X)
for iteration in range(iterations):
    # model predictions and error
    pred = X @ w + b
    error = pred -y
    # gradient descent 
    dw = (1/n)*(X.T@error)
    db = (1/n)*np.sum(error)
    # Updating parameter
    w = w-alpha*dw
    b = b-alpha*db

print(f"Final Weights: {w}")
print(f"Final Bias: {b}")
loss = np.mean((pred - y)**2)
print(f"Final Loss: {loss}")
