X = [1,2,3,4,5]
y = [7,11,15,19,23]
w = 0
b = 0
lr = 0.01
epochs = 1000
n = len(X)

for epoch in range(epochs):
    dw=0
    db=0
    total_loss = 0

    for i in range(n):
        y_pred = w*X[i] + b
        error = y[i] - y_pred
        total_loss += error **2
        dw += -2*X[i]*error
        db += -2 * error

    dw /= n
    db /= n
    loss = total_loss/n

    w = w-lr*dw
    b=b-lr*db
    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss, "w:", w, "b:", b)
print("Final w:", w)
print("Final b:", b)
print("Final loss:", loss)
