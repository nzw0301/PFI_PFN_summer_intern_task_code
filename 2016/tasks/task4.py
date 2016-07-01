from math import sqrt
import random

from task2 import affine
from task3 import train_on_batch


def glorot_uniform(shape):
    nb_next_units = shape[0]
    nb_pre_units = shape[1]

    upper = 1. / sqrt(nb_pre_units)

    W_n = [None] * nb_next_units
    for row in range(nb_next_units):
        W_n[row] = [random.uniform(-upper, upper) for _ in range(nb_pre_units)]

    b_n = [0. for _ in range(nb_next_units)]

    return W_n, b_n

def evaluate(x, W, b, eta=0.01):
    loss = 0.

    h = affine(x, W[0], b[0])
    y = affine(h, W[1], b[1])
    gy = [y_i - x_i for x_i, y_i in zip(x, y)]
    loss = sum([gy_i**2 for gy_i in gy]) / 2
    return loss

if __name__ == '__main__':
    random.seed(13)

    #  load data
    data_path = "../dataset.dat"
    f = open(data_path, "r")
    N, D = list(map(int, f.readline().split()))  # read header
    xs = [list(map(float, l.split())) for l in f.readlines()]
    f.close()

    # init weight params
    n = 5  # the number of hidden layer units
    W1, b1 = glorot_uniform((n, D))
    W2, b2 = glorot_uniform((D, n))

    W = [W1, W2]
    b = [b1, b2]

    # init trainig params
    eta = 0.001
    nb_epoch = 1000

    # training
    for epoch in range(1, nb_epoch):
        random.shuffle(xs)

        # training
        for x in xs:
            W, b, _ = train_on_batch(x, W, b, eta)

        # evaluation
        total_loss = 0.
        for x in xs:
            total_loss += evaluate(x, W, b, eta)
        average_loss = total_loss/len(xs)

        print("%2d" % epoch, "average loss:", average_loss)
        if average_loss <= 2.5 and epoch > 50:
            break

    # output weight params
    out_file = open("./../task4_weights.dat", "w")
    for W_n in W:
        for w in W_n:
            out_file.write(" ".join(map(str, w)))
            out_file.write("\n")

    for b_n in b:
        out_file.write(" ".join(map(str, b_n)))
        out_file.write("\n")

    out_file.close()
