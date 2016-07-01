def affine(x, W, b):
    result = [None] * len(b)
    for i, b_i in enumerate(b):
        result[i] = sum([W[i][j]*x_j for j, x_j in enumerate(x)]) + b_i
    return result

if __name__ == "__main__":
    # init test data
    x = [1, 2]

    W = [[1, 2],
         [-1, -2],
         [3, 4]]

    b = [3, 2, 1]

    result = affine(x, W, b)
    expected = [8, -3, 12]

    assert(result == expected)
