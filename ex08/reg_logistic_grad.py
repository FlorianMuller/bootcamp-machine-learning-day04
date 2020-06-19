import numpy as np
import functools


def regularized_grad(func):

    @functools.wraps(func)
    def wrapper(y, x, theta, lambda_, *args, **kwargs):
        ret_val = func(y, x, theta, lambda_, *args, **kwargs)

        if theta.ndim == 1:
            theta = theta[:, np.newaxis]

        if (theta.size == 0 or theta.ndim != 2 or theta.shape[1] != 1
                or ret_val is None):
            return None

        theta_prime = theta.copy()
        theta_prime[0] = 0

        return ret_val + (lambda_ * theta_prime / x.shape[0])

    return wrapper


@regularized_grad
def reg_logistic_grad(y, x, theta, lambda_):
    """
    Computes the regularized logistic gradient of three non-empty
    numpy.ndarray, with two for-loops. The three arrays must
    have compatible dimensions.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of dimension n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of dimension n * 1, containing the results
        of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if theta.ndim == 1:
        theta = theta[:, np.newaxis]

    if (x.size == 0 or y.size == 0 or theta.size == 0
        or x.ndim != 2 or y.ndim != 2 or theta.ndim != 2
            or x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]
            or y.shape[1] != 1 or theta.shape[1] != 1):
        return None

    x_prime = np.c_[np.ones(x.shape[0]), x]
    y_hat = 1 / (1 + np.exp(-x_prime @ theta))
    nabla = np.zeros(theta.shape)

    for j in range(theta.size):
        nabla[j] = np.sum(
            (y_hat - y) * x_prime[:, j].reshape(-1, 1)) / x.shape[0]

    return nabla


@regularized_grad
def vec_reg_logistic_grad(y, x, theta, lambda_):
    """
    Computes the regularized logistic gradient of three non-empty
    numpy.ndarray, without any for-loop. The three arrays must
    have compatible dimensions.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of dimension n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of dimension n * 1, containing the results
        of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if theta.ndim == 1:
        theta = theta[:, np.newaxis]

    if (x.size == 0 or y.size == 0 or theta.size == 0
        or x.ndim != 2 or y.ndim != 2 or theta.ndim != 2
            or x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]
            or y.shape[1] != 1 or theta.shape[1] != 1):
        return None

    x_prime = np.c_[np.ones(x.shape[0]), x]
    return (x_prime.T @ ((1 / (1 + np.exp(-x_prime @ theta))) - y)) / x.shape[0]


if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4],
                  [2, 4, 5, 5],
                  [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    # Example 1.1:
    print("1.1", reg_logistic_grad(y, x, theta, 1), end="\n\n")
    # Output:
    # array([[-0.55711039],
    # [-1.40334809],
    # [-1.91756886],
    # [-2.56737958],
    # [-3.03924017]])

    # Example 1.2:
    print("1.2", vec_reg_logistic_grad(y, x, theta, 1), end="\n\n")
    # Output:
    # array([[-0.55711039],
    # [-1.40334809],
    # [-1.91756886],
    # [-2.56737958],
    # [-3.03924017]])

    # Example 2.1:
    print("2.1", reg_logistic_grad(y, x, theta, 0.5), end="\n\n")
    # Output:
    # array([[-0.55711039],
    # [-1.15334809],
    # [-1.96756886],
    # [-2.33404624],
    # [-3.15590684]])

    # Example 2.2:
    print("2.2", vec_reg_logistic_grad(y, x, theta, 0.5), end="\n\n")
    # Output:
    # array([[-0.55711039],
    # [-1.15334809],
    # [-1.96756886],
    # [-2.33404624],
    # [-3.15590684]])

    # Example 3.1:
    print("3.1", reg_logistic_grad(y, x, theta, 0.0), end="\n\n")
    # Output:
    # array([[-0.55711039],
    # [-0.90334809],
    # [-2.01756886],
    # [-2.10071291],
    # [-3.27257351]])

    # Example 3.2:
    print("3.2", vec_reg_logistic_grad(y, x, theta, 0.0), end="\n\n")
    # Output:
    # array([[-0.55711039],
    # [-0.90334809],
    # [-2.01756886],
    # [-2.10071291],
    # [-3.27257351]])
