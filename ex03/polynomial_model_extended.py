import numpy as np


def add_polynomial_features(x, power):
    """
    Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power given in argument.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        power: has to be an int, the power up to which the columns of matrix x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of dimension m * (np), containg the polynomial feature values for all training examples.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    new_x = x
    for _ in range(power - 1):
        new_x = np.c_[new_x, new_x[:, :x.shape[1]] * new_x[:, -x.shape[1]:]]

    return new_x


if __name__ == "__main__":
    x = np.arange(1, 11).reshape(5, 2)

    # Example 1:
    print(add_polynomial_features(x, 3))
    # Output:
    # array([[1, 2, 1, 4, 1, 8],
    #        [3, 4, 9, 16, 27, 64],
    #        [5, 6, 25, 36, 125, 216],
    #        [7, 8, 49, 64, 343, 512],
    #        [9, 10, 81, 100, 729, 1000]])

    # Example 2:
    print(add_polynomial_features(x, 5))
    # Real Output:
    # array([[1, 2, 1, 4, 1, 8, 1, 16, 1, 32],
    #        [3, 4, 9, 16, 27, 64, 81, 256, 243, 1024],
    #        [5, 6, 25, 36, 125, 216, 625, 1296, 3125, 7776],
    #        [7, 8, 49, 64, 343, 512, 2401, 4096, 16807, 32768],
    #        [9, 10, 81, 100, 729, 1000, 6561, 10000, 59049, 100000]])

    # Output of the subject, (that is a power of 4 and not 5):
    # array([[1, 2, 1, 4, 1, 8, 1, 16],
    #        [3, 4, 9, 16, 27, 64, 81, 256],
    #        [5, 6, 25, 36, 125, 216, 625, 1296],
    #        [7, 8, 49, 64, 343, 512, 2401, 4096],
    #        [9, 10, 81, 100, 729, 1000, 6561, 10000]])
# Example 2:
add_polynomial_features(x, 5)
# Output:
array([[1, 2, 1, 4, 1, 8, 1, 16],
       [3, 4, 9, 16, 27, 64, 81, 256],
       [5, 6, 25, 36, 125, 216, 625, 1296],
       [7, 8, 49, 64, 343, 512, 2401, 4096],
       [9, 10, 81, 100, 729, 1000, 6561, 10000]])
