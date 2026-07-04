import numpy as np
X = np.array([
    [1,2],
    [2,1],
    [3,4],
    [4,6]
])
y = np.array([8,7,18,21])

X_train = X[:3]
y_train = y[:3]
X_test = X[3:]
y_test = y[3:]

w = np.zeros(2)
b = 0
lr = 0.01
epochs = 1000
n = len(X_train)
for epoch in range(epochs):
    y_pred = np.dot(X_train,w) +b
    error = y_train - y_pred
    train_loss = np.mean(error**2)
    dw = (-2/n)*np.dot(X_train.T,error)
    db = (-2/n)*np.sum(error)

    w = w-lr*dw
    b = b-lr*db


    ypred = np.dot(X_test,w) + b

    error = y_test - ypred

    loss_test = np.mean(error**2)

    if(epoch%100==0):
        print("Epoch:", epoch, "Loss:", train_loss)

print("Training w:", w)
print("Training b:", b)
print("Training loss:", train_loss)

X_test = X[3:]
y_test = y[3:]

ypred = np.dot(X_test,w) + b

error = y_test - ypred

loss_test = np.mean(error**2)

print(f'Test loss: {loss_test}')