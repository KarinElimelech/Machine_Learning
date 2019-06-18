
from random import shuffle

import numpy as np
from scipy.stats import mstats
import sys

'''
the function normlize acording to minmaxnrm
'''
def min_max_norm(xMatrix):
    colNum = len(xMatrix[0])
    for i in range(0, colNum):
        l = (xMatrix[:, [i]].max() - xMatrix[:, [i]].min())
        if (l == 0):
            return
        xMatrix[:, [i]] = (xMatrix[:, [i]] - xMatrix[:, [i]].min()) / l

'''
the function normlize acording to z-score
'''
def z_norm(xMatrix):
    # length of one row is number of columns
    colNum = len(xMatrix[0])
    for i in range(0, colNum):
        xMatrix[:, [i]] = mstats.zscore(xMatrix[:, [i]])

'''
the function training model according to perceptron algorithm
'''
def perceptron(xMatrix, yArray, w):
    epochs = 10
    eta = 0.1
    for e in range(epochs):
        c = list(zip(xMatrix, yArray))
        shuffle(c)
        xMatrix, yArray = zip(*c)
        # xMatrix, yArray = shuffle(xMatrix, yArray, random_state=1)
        for x, y in zip(xMatrix, yArray):
            # predict
            y_hat = np.argmax(np.dot(w, x))
            if (y != y_hat):
                w[y, :] = w[y, :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x
        if e > 7:
            eta /= 100

'''
the function training model according to SVM algorithm
'''
def svm(xMatrix, yArray, w):
    epochs = 10
    eta = 0.1
    lamda = 0.7
    for e in range(epochs):
        c = list(zip(xMatrix, yArray))
        shuffle(c)
        xMatrix, yArray = zip(*c)
        # xMatrix, yArray = shuffle(xMatrix, yArray, random_state=1)
        for x, y in zip(xMatrix, yArray):
            # predict
            y_hat = np.argmax(np.dot(w, x))
            if (y != y_hat):
                y_o = 0
                w[y, :] = (1 - lamda * eta) * w[y, :] + eta * x
                w[y_hat, :] = (1 - lamda * eta) * w[y_hat, :] - eta * x
                for l in range(3):
                    if l != y and l != y_hat:
                        y_o = l
                w[y_o, :] = (1 - lamda * eta) * w[y_o, :]
            else:
                for j in range(3):
                    w[j, :] = (1 - lamda * eta) * w[j, :]
        if e > 7:
            eta /= 100
            lamda/=100

'''
the function training model according to PA algorithm
'''
def pa(xMatrix, yArray, w):
    epochs = 10
    for e in range(epochs):
        c = list(zip(xMatrix, yArray))
        shuffle(c)
        xMatrix, yArray = zip(*c)
        for x, y in zip(xMatrix, yArray):
            # predict
            y_hat = np.argmax(np.dot(w, x))
            if (y != y_hat):
                tau = computeTau(x, y, y_hat, w)
                w[y, :] = w[y, :] + tau * x
                w[y_hat, :] = w[y_hat, :] - tau * x

'''
the function gets x,y,y_hat,w
and compute tau and hinge loss
'''
def computeTau(x, y, y_hat, w):
    w_yHat_x = np.dot(w[y_hat,:],x)
    w_y_x = np.dot(w[y,:],x)
    l = 1 - w_y_x + w_yHat_x
    loss = max(0, l)
    demon = 2*pow(np.linalg.norm(x),2)
    tau = loss // demon
    return tau

'''
help function that compare prediction to label and compute the loss
'''
def compute_loss(x_test, y_test, w, algo):
    ftest = 0
    times = 50
    for t in range(times):
        loss = 0
        m = len(x_test)
        for i in range(0, m):
            # ran = random.randint(0,3000)
            y_hat = np.argmax(np.dot(w, x_test[i]))
            if y_test[i] != y_hat:
                loss += 1
        test = float(loss / m)
        ftest += test
    ftest /= times
    print(algo, "err= ", ftest)

'''
the function test new data
'''
def test(x_test, w1, w2, w3):
    m = len(x_test)
    for i in range(0, m):
        # ran = random.randint(0,3000)
        perc_y = np.argmax(np.dot(w1, x_test[i]))
        svm_y = np.argmax(np.dot(w2, x_test[i]))
        pa_y = np.argmax(np.dot(w3, x_test[i]))
        print("perceptron: %s, svm: %s, pa: %s" % (perc_y, svm_y, pa_y))

'''
the main function - get args create matrix and arr and train models
and test new data.
'''
def main():
    train_x = sys.argv[1]
    train_y = sys.argv[2]
    # generating string array from train_x
    xMatrix = (np.genfromtxt(train_x, dtype='str', delimiter=","))
    # converting sex to number
    xMatrix = np.char.replace(xMatrix, 'M', '0')
    xMatrix = np.char.replace(xMatrix, 'F', '1')
    xMatrix = np.char.replace(xMatrix, 'I', '2')
    # converting strings to floats
    xMatrix = xMatrix.astype(np.float)
    # generating float array from train_y
    yArray = (np.genfromtxt(train_y, dtype='float', delimiter=","))
    # converting train_y to array of int
    yArray = yArray.astype(np.int)
    # normalizing xMatrix
    min_max_norm(xMatrix)
    #z_norm(xMatrix)
    w1 = np.zeros((3, 8), dtype=float)
    w2 = np.zeros((3, 8), dtype=float)
    w3 = np.zeros((3, 8), dtype=float)
    # RUNNING ALGORITHMS
    perceptron(xMatrix, yArray, w1)
    svm(xMatrix, yArray, w2)
    pa(xMatrix, yArray, w3)
    # TEST
    test_x = sys.argv[3]
    # generating string array from train_x
    xtest = (np.genfromtxt(test_x, dtype='str', delimiter=","))
    # converting sex to number
    xtest = np.char.replace(xtest, 'M', '0')
    xtest = np.char.replace(xtest, 'F', '1')
    xtest = np.char.replace(xtest, 'I', '2')
    # converting strings to floats
    xtest = xtest.astype(np.float)
    '''
    #if you want to compute the loss - delete note
    test_y = sys.argv[4]
    #generating float array from train_y
    ytest = (np.genfromtxt(test_y, dtype='float', delimiter=","))
    #converting train_y to array of int
    ytest = ytest.astype(np.int)
    compute_loss(xtest, ytest, w1, "perc")
    compute_loss(xtest, ytest, w2, "svm")
    compute_loss(xtest, ytest, w3, "pa")
    '''
    test(xtest, w1, w2, w3)
    return 1


if __name__ == '__main__':
    main()
