import numpy as np
def feature_normalizing(X):
    mean = np.mean(X,axis=0) # taking mean along column
    std = np.std(X,axis=0) # taking standard deveation along column

    return (X-mean) / std 

no_of_hours = np.random.randint(100,300,300)
no_of_subjects = np.random.randint(1,6,300)
days = np.random.randint(0,7,300)

X = np.column_stack((no_of_hours,no_of_subjects,days))
norm_X = feature_normalizing(X)
noise = np.random.randn(300) *5
y = 0.4*no_of_hours + 5*no_of_subjects - 2*days + noise
w = np.random.randn(3)
b = 0
lr = 0.01
epochs = 2000
n = len(norm_X)

for epoch in range(epochs):
    y_pred = np.dot(norm_X,w) + b
    loss = np.mean((y-y_pred)**2)
    dw = (-2/n)*np.dot(norm_X.T, y-y_pred)
    db = (-2/n)*np.sum(y-y_pred)

    w = w - lr * dw
    b = b - lr * db

print("Final w:", w)
print("Final b:", b)
print("Final loss:", loss)
