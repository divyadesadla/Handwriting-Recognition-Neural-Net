import math
import csv
import sys
import numpy as np


def labels(Data):
    label = Data[:, 0]
    label = np.asarray(label, dtype='int')
    return label


def labels_onehot(label):
    num_rows = len(label)
    num_col = 10

    a = np.zeros((num_rows, num_col))
    a[np.arange(num_rows), label] = 1
    return a


def gen_inputs(Data):
    s = np.shape(Data)
    if len(s) == 1:
        data_bias = np.ones(len(Data) + 1)
        data_bias[1:] = Data
    else:
        rows, cols = s
        data_bias = np.ones((rows, cols+1))
        data_bias[:, 1:] = Data

    return data_bias

# #################################################################


def initialization_random(inputs_train, hidden_units):
    num_inputs = len(inputs_train[0])
    num_hidden_units = hidden_units
    num_outputs = 10

    random_alpha = np.random.uniform(
        low=-0.1, high=0.1, size=(num_hidden_units, num_inputs))
    random_alpha[:, 0] = 1
    random_beta = np.random.uniform(
        low=-0.1, high=0.1, size=(num_outputs, hidden_units+1))
    random_beta[:, 0] = 1
    # print(np.shape(random_beta))
    return random_alpha, random_beta


def initialization_zeros(inputs_train, hidden_units):
    num_inputs = len(inputs_train[0])
    num_hidden_units = hidden_units
    num_outputs = 10

    zeros_alpha = np.zeros((num_hidden_units, num_inputs))
    # zeros_alpha[:, 0] = 1
    zeros_beta = np.zeros((num_outputs, hidden_units+1))
    # zeros_beta[:, 0] = 1
    return zeros_alpha, zeros_beta

# #################################################################


def LinearForward(data, weights):
    out = np.dot(weights, data.T)
    return out


def SigmoidForward(a):
    z_denom = 1 + np.exp(-a)
    z = 1 / z_denom
    return z


def Softmax(b):
    k = np.exp(b)
    y = k / np.sum(k, axis=0)
    return y


def Cross_Entropy(one_hot, y):
    y_log = np.log(y.T)
    J = - np.multiply(one_hot, y_log)

    if len(np.shape(J)) == 1:
        J_out = np.sum(J)
    else:
        J_out = np.sum(J, axis=1)
    J_out = np.mean(J_out)
    return J_out


# #################################################################

def LinearBackward(a, weights, b, g_b):
    g_w = np.reshape(g_b, (-1, 1)) * np.reshape(a, (1, -1))
    g_a = np.dot(g_b, weights)
    g_a = g_a[1:]
    return g_w, g_a


def SoftmaxBackward(one_hot, y):
    g_b = y.T - one_hot
    return g_b


def SigmoidBackward(g_z, z):
    g_a1 = np.multiply(z, 1-z)
    g_a = np.multiply(g_z, g_a1)
    return g_a

# ###############################################################


def NNForward(inputs_train, alpha, beta, labels_onehot):
    a = LinearForward(inputs_train, alpha)
    z = SigmoidForward(a)
    z_bias = gen_inputs(z.T)
    b = LinearForward(z_bias, beta)
    y = Softmax(b)
    J = Cross_Entropy(labels_onehot, y)
    return (a, z, z_bias, b, y, J)


def NNBackward(inputs_train, alpha, beta, labels_onehot, o):
    a, z, z_bias, b, y, J = o
    g_b = SoftmaxBackward(labels_onehot, y)
    g_beta, g_z = LinearBackward(z_bias, beta, b, g_b)
    g_a = SigmoidBackward(g_z, z)
    g_alpha, g_x = LinearBackward(inputs_train, alpha, a, g_a)
    return (g_b, g_beta, g_z, g_a, g_alpha, g_x)


def get_new_weights(alpha, learning_rate, g_alpha, beta, g_beta):
    alpha_new = alpha - (learning_rate * g_alpha)
    beta_new = beta - (learning_rate * g_beta)
    return alpha_new, beta_new


# ###############################################################

def train_nn(train_input, one_hot_train, alpha, beta, learning_rate):
    for i in range(len(train_input)):
        F_out = NNForward(train_input[i], alpha, beta, one_hot_train[i])
        B_out = NNBackward(
            train_input[i], alpha, beta, one_hot_train[i], F_out)
        alpha, beta = get_new_weights(
            alpha, learning_rate, B_out[4], beta, B_out[1])
    return alpha, beta


def test_nn(test_input, one_hot_test, alpha, beta):
    a, z, z_bias, b, y, J = NNForward(test_input, alpha, beta, one_hot_test)
    return J


def run_NN(inputs_train, inputs_test, labels_onehot_train, labels_onehot_test, alpha, beta, learning_rate, num_epoch):
    results = []
    for i in range(num_epoch):
        alpha, beta = train_nn(
            inputs_train, labels_onehot_train, alpha, beta, learning_rate)
        J_train = test_nn(inputs_train, labels_onehot_train, alpha, beta)
        J_test = test_nn(inputs_test, labels_onehot_test, alpha, beta)
        results.append('epoch={} crossentropy(train): {}'.format(i+1, J_train))
        results.append('epoch={} crossentropy(test): {}'.format(i+1, J_test))
        # print('Train = {}, Test = {}'.format(J_train, J_test))
    return alpha, beta, results


# ###############################################################

def prediction(Data, alpha, beta, labels_onehot):
    l = []
    for i in range(len(Data)):
        l.append(NNForward(Data[i], alpha, beta, labels_onehot[i])[4])

    y_predict = []
    for k in l:
        y_predict.append(np.argmax(k))
    return y_predict


def error(y, y_predict):
    Total = len(y)
    count = 0

    for i in range(Total):
        if np.argmax(y[i]) != y_predict[i]:
            count = count + 1
    error = count/Total
    return error


if __name__ == "__main__":
    # x = np.array([1, 1, 1, 0, 0, 1, 1])
    # alpha = np.array([[1, 1, 2, -3, 0, 1, -3], [1, 3, 1, 2, 1, 0, 2],
    #                   [1, 2, 2, 2, 2, 2, 1], [1, 1, 0, 2, 1, -2, 2]])
    # beta = np.array([[1, 1, 2, -2, 1], [1, 1, -1, 1, 2], [1, 3, 1, -1, 1]])
    # hidden_units = 4
    # a = LinearForward(x, alpha)
    # z = SigmoidForward(a)
    # z_bias = gen_inputs(z)
    # b = LinearForward(z_bias, beta)
    # y = Softmax(b)
    # print('a = {}'.format(a))
    # print('z = {}'.format(z))
    # print('b = {}'.format(b))
    # print('y = {}'.format(y))
    # exit(0)

    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    # num_epoch = 2
    # hidden_units = 4
    # init_flag = 2
    # learning_rate = 0.1

    train_open = open(train_input, 'r')
    test_open = open(test_input, 'r')

    train_read = csv.reader(train_open)
    test_read = csv.reader(test_open)

    data = list()
    for i in train_read:
        data.append(i)

    data = np.asarray(data)

    testdata = list()
    for i in test_read:
        testdata.append(i)

    testdata = np.asarray(testdata)

    labels_train = labels(data)
    labels_onehot_train = labels_onehot(labels_train)
    inputs_train = gen_inputs(data[:, 1:])

    labels_test = labels(testdata)
    labels_onehot_test = labels_onehot(labels_test)
    inputs_test = gen_inputs(testdata[:, 1:])

    if init_flag == 1:
        alpha, beta = initialization_random(inputs_train, hidden_units)
    elif init_flag == 2:
        alpha, beta = initialization_zeros(inputs_train, hidden_units)

    alpha, beta, results = run_NN(inputs_train, inputs_test, labels_onehot_train,
                                  labels_onehot_test, alpha, beta, learning_rate, num_epoch)

    y_train = prediction(inputs_train, alpha, beta, labels_onehot_train)
    y_test = prediction(inputs_test, alpha, beta, labels_onehot_test)

    error_train = error(labels_onehot_train, y_train)
    # print(error_train)
    error_test = error(labels_onehot_test, y_test)
    # print(error_test)

    train_open_write = open(train_out, 'w')
    train_outputz = ''
    for k in y_train:
        train_outputz += str(k) + '\n'
    train_open_write.write(str(train_outputz))

    test_open_write = open(test_out, 'w')
    test_outputz = ''
    for k in y_test:
        test_outputz += str(k) + '\n'
    test_open_write.write(str(test_outputz))

    ans1 = 'error(train): ' + str(error_train)
    ans2 = 'error(test): ' + str(error_test)

    metrics_out_file = open(metrics_out, 'w')
    metrics_outputz = ''
    for k in results:
        metrics_outputz += str(k) + '\n'
    metrics_outputz += ans1 + '\n' + ans2 + '\n'
    metrics_out_file.write(metrics_outputz)
