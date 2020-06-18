import numpy as np


def iterative_l2(theta):
    """
    Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if theta.size == 0:
        return None

    # With np.sum
    return np.sum(theta[1:] ** 2)

    # With a real foor loop
    # res = 0
    # for t in theta[1:]:
    #     res += t ** 2
    # return res


def l2(theta):
    """
    Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    # Using one dimensional array to use dot product with np.dot
    # (np.dot use matmul with two dimensional array)
    if theta.ndim == 2 and theta.shape[1] == 1:
        theta = theta.flatten()

    if theta.size == 0 or theta.ndim != 1:
        return None

    theta_prime = theta[1:]

    return np.dot(theta_prime, theta_prime)


if __name__ == "__main__":
    x = np.array([2, 14, -13, 5, 12, 4, -19])

    # Example 1:
    print(iterative_l2(x))
    # Output:
    # 911.0

    # Example 2:
    print(l2(x))
    # Output:
    # 911.0
    y = np.array([3, 0.5, -6])

    # Example 3:
    print(iterative_l2(y))
    # Output:
    # 36.25

    # Example 4:
    print(l2(y))
    # Output:
    # 36.25
