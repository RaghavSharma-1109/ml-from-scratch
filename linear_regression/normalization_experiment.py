import numpy as np

X = np.array([
[1, 1000],
[2, 2000],
[3, 3000],
[4, 4000],
[5, 5000]
])

y = np.array([10,20,30,40,50])
# Without Normaloization
w = np.random.randn(2)
b = 0
lr = 0.03
epochs = 1000
for epoch in range(epochs):
    pred = X@w + b
    error = pred - y

    dw = (2/len(X))*(X.T@error)
    db = (2/len(X))*(np.sum(error))

    w = w - lr*dw
    b = b - lr*db

without_norm_final_w = w
without_norm_final_b = b

# With Normalization
X_norm = (X-np.mean(X,axis=0))/ np.std(X,axis=0)
w = np.random.randn(2)
b = 0

for epoch in range(epochs):
    pred = X_norm@w + b
    error = pred - y

    dw = (2/len(X_norm))*(X_norm.T@error)
    db = (2/len(X_norm))*np.sum(error)

    w = w-lr*dw
    b = b-lr*db
with_norm_final_w = w
with_norm_final_b = b
loss_without = np.mean((X@without_norm_final_w + without_norm_final_b - y)**2)
loss_with = np.mean((X_norm@with_norm_final_w + with_norm_final_b - y)**2)


#Comparison 
print("Loss without normalization:", loss_without)
print("Loss with normalization:", loss_with)
print(f"Without Normalization: 1) Final w= {without_norm_final_w } 2) Final b= {without_norm_final_b}")
print(f"With Normalization: 1) Final w= {with_norm_final_w} 2) Final b= {with_norm_final_b} ")

# nan-> gradients exploaded due to the different scale 
# Without Normalization: 1) Final w= [nan nan] 2) Final b= nan
# With Normalization: 1) Final w= [7.35014427 6.79199135] 2) Final b= 29.99999999999997
