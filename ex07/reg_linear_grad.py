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
def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty
    numpy.ndarray, with two for-loop. The three arrays must have
    compatible dimensions.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of dimension n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of dimension n * 1, containing the
        results of the formula for all j.
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
    y_hat = x_prime @ theta
    nabla = np.zeros(theta.shape)

    for j in range(theta.size):
        nabla[j] = np.sum(
            (y_hat - y) * x_prime[:, j].reshape(-1, 1)) / x.shape[0]

    return nabla


@regularized_grad
def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty
    numpy.ndarray, without any for-loop. The three arrays must have
    compatible dimensions.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of dimension n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of dimension n * 1, containing the
        results of the formula for all j.
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
    return (x_prime.T @ ((x_prime @ theta) - y)) / x.shape[0]


if __name__ == "__main__":
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])

    # Example 1.1:
    print("1.1", reg_linear_grad(y, x, theta, 1), end="\n\n")
    # Output:
    # array([[ -60.99 ],
    # [-195.64714286],
    # [ 863.46571429],
    # [-644.52142857]])

    # Example 1.2:
    print("1.2", vec_reg_linear_grad(y, x, theta, 1), end="\n\n")
    # Output:
    # array([[ -60.99 ],
    # [-195.64714286],
    # [ 863.46571429],
    # [-644.52142857]])

    # Example 2.1:
    print("2.1", reg_linear_grad(y, x, theta, 0.5), end="\n\n")
    # Output:
    # array([[ -60.99 ],
    # [-195.86142857],
    # [ 862.71571429],
    # [-644.09285714]])

    # Example 2.2:
    print("2.2", vec_reg_linear_grad(y, x, theta, 0.5), end="\n\n")
    # Output:
    # array([[ -60.99 ],
    # [-195.86142857],
    # [ 862.71571429],
    # [-644.09285714]])

    # Example 3.1:
    print("3.1", reg_linear_grad(y, x, theta, 0.0), end="\n\n")
    # Output:
    # array([[ -60.99 ],
    # [-196.07571429],
    # [ 861.96571429],
    # [-643.66428571]])

    # Example 3.2:
    print("3.2", vec_reg_linear_grad(y, x, theta, 0.0), end="\n\n")
    # Output:
    # array([[ -60.99 ],
    # [-196.07571429],
    # [ 861.96571429],
    # [-643.66428571]])
