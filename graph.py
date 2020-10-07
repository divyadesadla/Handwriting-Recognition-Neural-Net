import numpy as np
import csv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # hidden_units = [5, 20, 50, 100, 200]
    learning_rate = [0.1, 0.01, 0.001]

    for i in learning_rate:
        J_train = []
        J_test = []
        file = 'out_metrics_{}.txt'.format(i)
        with open(file, 'r') as f:
            data = f.read()

        data = data.split('\n')
        data = np.asarray(data)
        for j in range(len(data)-3):
            a = float(data[j].strip().split(':')[1].strip())
            if j % 2 == 0:
                J_train.append(a)
            else:
                J_test.append(a)
        # print(data.shape)
        # print(data[-5:])
        # a = data[-5].split(':')[1].strip()
        # b = data[-4].split(':')[1].strip()
        # print(a)
        # print(b)
        # J_train.append(float(a))
        # J_test.append(float(b))

        fig = plt.figure()
        plt.plot(range(1, 101), J_train, label='Train Data')
        plt.plot(range(1, 101), J_test, label='Test Data')
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Average Cross Entropy')
        plt.legend(loc='best')
        plt.savefig('learning_rate_{}.png'.format(i))
