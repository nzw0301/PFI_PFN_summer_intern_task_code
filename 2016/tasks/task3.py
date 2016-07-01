from task1 import outer
from task2 import affine


def train_on_batch(x, W, b, eta=0.01):
    # forward
    h = affine(x, W[0], b[0])
    y = affine(h, W[1], b[1])
    gy = [y_i - x_i for x_i, y_i in zip(x, y)]

    loss = sum([gy_i**2 for gy_i in gy]) / 2

    # backward
    ## gW2
    gW_2 = outer(gy, h)

    ## gW1
    gh = [sum( W[1][j][i]*gy_i for j, gy_i in enumerate(gy)) for i in range(len(W[1][0]))]
    gW_1 = outer(gh, x)

    # update
    ## weights
    for i in range(len(W[1])):
        for j in range(len(W[1][i])):
            W[1][i][j] -= (eta*gW_2[i][j])

    for i in range(len(W[0])):
        for j in range(len(W[0][i])):
            W[0][i][j] -= (eta*gW_1[i][j])

    ## bias
    b[1] = [b[1][i] - eta*gy_i for i, gy_i in enumerate(gy)]
    b[0] = [b[0][i] - eta*gh_i for i, gh_i in enumerate(gh)]

    return W, b, loss


if __name__ == "__main__":
    from unittest import TestCase

    # init test data
    xs = [[ 1., -1.]]
    b =  [[ 1.,  2., 3.],
          [-1., -2.]]

    W1 = [[1.,  2.],
          [0.,  1.],
          [1., -1.]]

    W2 = [[ 1., 2.,  1.],
          [-1., 2., -1.]]
    W = [W1, W2]

    loss = 0.
    for x in xs:
        W, b, loss = train_on_batch(x, W, b)

    W1, W2 = W
    b1, b2 = b

    # test loss
    expected_loss = 20.5
    TestCase.assertAlmostEquals(TestCase, first=loss, second=expected_loss)

    # test W2
    expected_W2 = [[ 1., 1.95,  0.75],
                   [-1., 2.04, -0.8]]
    for column in range(len(W2)):
        for row in range(len(W2[column])):
            TestCase.assertAlmostEquals(TestCase, first=W2[column][row], second=expected_W2[column][row])

    # test W1
    expected_W1 = [[ 0.91,  2.09],
                   [-0.02,  1.02],
                   [ 0.91, -0.91]]
    for column in range(len(W1)):
        for row in range(len(W1[column])):
            TestCase.assertAlmostEquals(TestCase, first=W1[column][row], second=expected_W1[column][row])

    # test b2
    expected_b2 = [-1.05, -1.96]
    for row in range(len(b2)):
        TestCase.assertAlmostEquals(TestCase, first=b2[row] , second=expected_b2[row])

    # test b1
    expected_b1 = [0.91, 1.98, 2.91]
    for row in range(len(b1)):
        TestCase.assertAlmostEquals(TestCase, first=b1[row], second=expected_b1[row])
