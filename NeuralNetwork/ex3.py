
import numpy as np

input_size, hidden_layers, hidden_size, output_size = 28 * 28, 1, 80, 10
norm = 255

'''
compute epsilone according to parameters
'''
def epsi(n, m):
    return np.sqrt(6 / (n + m))

'''
NLL loss function
'''
def Loss(predict, real):
    y = (np.eye(output_size)[int(real)])
    ans = -np.dot(y, np.log(predict + 1.e-8))
    return ans

'''
create new matrix with random numbers in range of epsi
'''
def initialize_mat(row, col, epsi):
    return np.random.uniform(-epsi, epsi, (row, col))

'''
ReLU active function
'''
def relu(x):
    return np.maximum(x, 0)

'''
compute soft max on x
'''
def softmax(x):
    ans = np.exp(x) / sum(np.exp(x))
    ans[ans == 0] = 0.000000001
    return ans

'''
the function compute the network forward
according to the input
'''
def forward(params, active_func, x):
    w1 = params[0]
    b1 = params[1]
    w2 = params[2]
    b2 = params[3]
    x = np.array(x)
    x.shape = (input_size, 1)
    z1 = np.dot(w1, x)
    z1 = np.add(z1, b1)
    h1 = active_func(z1)
    z2 = np.dot(w2, h1)
    z2 = np.add(z2, b2)
    # z2 = np.divide(z2, len(x))
    y_hat = softmax(z2)
    return {'w2': w2, 'z1': z1, 'h1': h1, 'z2': z2, 'y_hat': y_hat, 'x': x}

'''
compute gradients to the back propagation
'''
def back_propagation(fprop, y, active_func_der):
    y = (np.eye(output_size)[int(y)]).reshape(-1, 1)
    w2, z1, h1, z2, y_hat, x = [fprop[key] for key in ('w2', 'z1', 'h1', 'z2', 'y_hat', 'x')]
    dz2 = y_hat - y  # dL/dz2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.multiply(np.dot(w2.T, dz2), active_func_der(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'db1': db1, 'dw1': dW1, 'db2': db2, 'dw2': dW2}

'''
update the weights with sgd
'''
def update_weights_sgd(params, gradients, lr):
    w1, b1, w2, b2 = params[0], params[1], params[2], params[3]
    params[0] = w1 - np.multiply(lr, gradients['dw1'])
    params[1] = b1 - np.multiply(lr, gradients['db1'])
    params[2] = w2 - np.multiply(lr, gradients['dw2'])
    params[3] = b2 - np.multiply(lr, gradients['db2'])
    return params

'''
the der of ReLU
'''
def relu_der(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

'''
compute loss and validation
'''
def predict_on_dev(params, active_func, dev_x, dev_y):
    sum_loss = good = 0.0  # good counts how many times we were correct
    for x, y in zip(dev_x, dev_y):
        # get probabilities vector as result, where index y is the probability that x is classified as tag y
        out = forward(params, active_func, x)
        # compute loss to see train loss (for hyper parameters tuning)
        loss = Loss(out['y_hat'], y)
        sum_loss += loss
        # model was correct
        if out['y_hat'].argmax() == y:
            good += 1
    # how many times we were correct / # of examples
    acc = good / dev_x.shape[0]
    avg_loss = sum_loss / dev_x.shape[0]  # avg. loss
    return avg_loss, acc

'''
run in loop on #epoche and compute predect,loss, BP and update
'''
def train(params, epochs, active_func, active_func_der, lr, train_x, train_y, dev_x, dev_y):
    t_size = train_x.shape[0]
    for i in range(epochs):
        sum_loss = 0.0
        # shuffle train examples - helps the model to learn (won't just remember order
        # $ shuffle(train_x, train_y)
        c = list(zip(train_x, train_y))
        np.random.shuffle(c)
        train_x, train_y = zip(*c)
        for x, y in zip(train_x, train_y):
            # get probabilities vector as result,
            # where index y is the probability that x is classified as ta1g y
            fprop = forward(params, active_func, x)
            # compute loss to see train loss (for hyper parameters tuning)
            loss = Loss(fprop['y_hat'], y)
            sum_loss += loss
            # returns the gradients for each parameter
            gradients = back_propagation(fprop, y, active_func_der)
            # updates the weights
            params = update_weights_sgd(params, gradients, lr)
            # after each epoch, check accuracy and loss on dev set for hyper parameter
            # tuning
        dev_loss, acc = predict_on_dev(params, active_func, dev_x, dev_y)
        #print(i, sum_loss / t_size, dev_loss, "{}%".format(acc * 100))

'''
the test set - call forward and save the predict to file
'''
def test(test_x, params, active_func):
    f = open("test_y", "w")
    for x in test_x:
        out = forward(params, active_func, x)
        f.write(str(out['y_hat'].argmax()) + "\n")
    f.close()

'''
the main function - 
get data, create w and b, and train model
'''
def main():
    train_x = np.loadtxt("train_x") / norm
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x") / norm
    train_size = len(train_x)
    e1 = epsi(hidden_size, input_size)
    w1, b1 = initialize_mat(hidden_size, input_size, e1), initialize_mat(input_size, 1, e1)
    e2 = epsi(output_size, hidden_size)
    w2, b2 = initialize_mat(output_size, hidden_size, e2), initialize_mat(output_size, 1, e2)
    c = list(zip(w1, b1))
    np.random.shuffle(c)
    w1, b1 = zip(*c)
    params = [w1, b1, w2, b2]
    dev_size = train_size * 0.2
    dev_size = int(dev_size)
    dev_x = train_x[-dev_size:, :]
    dev_y = train_y[-dev_size:]
    train_x, train_y = train_x[:-dev_size, :], train_y[:-dev_size]
    epoch = 25
    lr = 0.01
    train(params, epoch, relu, relu_der, lr, train_x, train_y, dev_x, dev_y)
    test(test_x, params, relu)


if __name__ == '__main__':
    main()
