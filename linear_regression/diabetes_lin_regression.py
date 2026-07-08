import random
import numpy as np
from sklearn.datasets import load_diabetes # gives a real data of 442 patients and 10 features(age, BMI etc)

data = load_diabetes() 
X= data.data # shape (442,10)
y = data.target # shape (442,)- 442 target values, one per row.
original_x10 = X[10].copy()
original_y10=y[10]

# shufftling the data:

shuffeled_indices = np.arange(442)
np.random.shuffle(shuffeled_indices)

X = X[shuffeled_indices]
y=y[shuffeled_indices]

position = np.where(shuffeled_indices == 10)

if original_y10 == y[position]:
    print("True")
else:
    print("False")

print(np.array_equal(original_x10, X[position].flatten()))
X_train = X[:350]
y_train = y[:350]
X_test = X[350:]
y_test = y[350:]

w = np.zeros(10)
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

    if(epoch%100==0):
        print("Epoch:", epoch, "Loss:", train_loss)


y_pred = np.dot(X_test,w) +b
error = y_test - y_pred
test_loss = np.mean(error**2)

baseline_pred = np.mean(y_train)
baseline_loss = np.mean((y_test - baseline_pred)**2)


print(" Test Loss:", test_loss)
print("Baseline Loss: ", baseline_loss)