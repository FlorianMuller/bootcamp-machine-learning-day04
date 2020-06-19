import numpy as np
import functools


def regularized(func):

    @functools.wraps(func)
    def wrapper(y, y_hat, theta, lambda_, *args, **kwargs):
        ret_val = func(y, y_hat, theta, lambda_, *args, **kwargs)

        # Using one dimensional array to use dot product with np.dot
        # (np.dot use matmul with two dimensional array)
        if theta.ndim == 2 and theta.shape[1] == 1:
            theta = theta.flatten()

        if theta.size == 0 or theta.ndim != 1 or ret_val is None:
            return None

        theta_prime = theta[1:]
        return ret_val + (lambda_ * np.dot(theta_prime, theta_prime)) / (2 * y.shape[0])

    return wrapper


@regularized
def reg_log_cost_(y, y_hat, theta, lambda_, eps=1e-15):
    """
    Computes the regularized cost of a logistic regression model from two non-empty numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be a numpy.ndarray, a vector of dimension n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized cost as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same dimensions.
    Raises:
        This function should not raise any Exception.
    """
    # Using one dimensional array to use dot product with np.dot
    # (np.dot use matmul with two dimensional array)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    if y_hat.ndim == 2 and y_hat.shape[1] == 1:
        y_hat = y_hat.flatten()

    if (y.size == 0 or y_hat.size == 0
        or y.ndim != 1 or y_hat.ndim != 1
            or y.shape != y_hat.shape):
        return None

    return -(y.dot(np.log(y_hat + eps)) + (1 - y).dot(np.log(1 - y_hat + eps))) / y.shape[0]


if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 1, 0])
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01])
    theta = np.array([1, 2.5, 1.5, -0.9])

    # Example :
    print(reg_log_cost_(y, y_hat, theta, .5))
    # Output:
    # 0.43377043716475955

    # Example :
    print(reg_log_cost_(y, y_hat, theta, .05))
    # Output:
    # 0.13452043716475953

    # Example :
    print(reg_log_cost_(y, y_hat, theta, .9))
    # Output:
    # 0.6997704371647596
