import numpy as np


def add_polynomial_features(x, power):
    """
    Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of dimension m * n, containg he polynomial feature values for all training examples.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    new_x = x
    for _ in range(power - 1):
        new_x = np.c_[new_x, new_x[:, -1] * new_x[:, 0]]

    return new_x


if __name__ == "__main__":
    x = np.arange(1, 6).reshape(-1, 1)

    # Example 1:
    print(add_polynomial_features(x, 3))
    # Output:
    # array([[ 1, 1, 1],
    # [ 2, 4, 8],
    # [ 3, 9, 27],
    # [ 4, 16, 64],
    # [ 5, 25, 125]])

    # Example 2:
    print(add_polynomial_features(x, 6))
    # Output:
    # array([[ 1, 1, 1, 1, 1, 1],
    # [ 2, 4, 8, 16, 32, 64],
    # [ 3, 9, 27, 81, 243, 729],
    # [ 4, 16, 64, 256, 1024, 4096],
    # [ 5, 25, 125, 625, 3125, 15625]])
