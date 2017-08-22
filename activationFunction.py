import numpy as np

def leakyReLU( x, leakyRate ):

    alpha = leakyRate  ##when x<0, output alpha*x
    x_abs = np.abs(x)
    y = (1 - alpha)/2 * x_abs + (1 + alpha)/2 * x

    return y

def leakyReLUGradient( x, leakyRate ):

    alpha = leakyRate
    x_abs = np.abs(x)
    g = (1 + alpha)/2 * 1 + (x/x_abs) * (1 - alpha)/2

    return g

def sigmoid( x ):

    y = 1/(1 + np.exp(-x))

    return y

def sigmoidGradient( x ):

    if isinstance(x, np.matrix):
        print('Error in sigmoidGradient: can\'t support matrix type\n')
        exit(1)# todo throw error

    g = sigmoid(x) * (1 - sigmoid(x))

    return g


if __name__ == '__main__':
    x = np.array([[[-0.25435214, 0.14849252, 0.28359378, 0.54253786],
                   [0.11925088, -0.14849252, 0.28359378, 0.54253786],
                   [0.11925088, 0.25435214, -0.28359378, 0.54253786]],
                  [[0.06751567, -0.11925088, 0.25435214, 0.54253786],
                   [-0.06751567, 0.11925088, 0.25435214, 0.28359378],
                   [0.06751567, 0.11925088, 0.25435214, -0.14849252]]])

    print(leakyReLU(x / x.__abs__(), 0.4))
    print(leakyReLUGradient(x, 0.4))
    print(sigmoid(x / x.__abs__()))
    print(sigmoidGradient(x / x.__abs__()))

    y = np.matrix('1.0 2.0; 3.0 4.0; 5.0 6.0')
    print(y)
    # print(sigmoidGradient(y))
    print(sigmoid(y))



