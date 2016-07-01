def outer(x, y):
    return [[x_i*y_j for y_j in y] for x_i in x]


if __name__ == "__main__":
    # init test data
    x = [1, 2, 3]
    y = [3, 2, 1, 0]

    result = outer(x, y)
    expected = [[3, 2, 1, 0],
                [6, 4, 2, 0],
                [9, 6, 3, 0]]

    assert (result == expected)
