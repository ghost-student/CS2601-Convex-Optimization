import numpy as np


def newton(fp, fpp, x0, tol=1e-5, maxiter=100000):
    """
    fp: function that takes an input x and returns the gradient of f at x
    fpp: function that takes an input x and returns the Hessian of f at x
    x0: initial point
    tol: toleracne parameter in the stopping crieterion. Newton's method stops
         when the 2-norm of the gradient is smaller than tol
    maxiter: maximum number of iterations

    This function should return a list of the sequence of approximate solutions
    x_k produced by each iteration
    """
    x_traces = [x0]
    x = x0
    #   START OF YOUR CODE
    while np.linalg.norm(fp(x)) > tol and len(x_traces) <= maxiter:
        x -= 1 / fpp(x) * fp(x)
        x_traces.append(x)
    #    END OF YOUR CODE
    return x_traces


def newton_damped(f, fp, fpp, x0, initial_stepsize=1.0, alpha=0.1, beta=0.7, tol=1e-5, maxiter=100000):
    """
    f: function that takes an input x an returns the value of f at x
    fp: function that takes an input x and returns the gradient of f at x
    fpp: function that takes an input x and returns the Hessian of f at x
    x0: initial point in gradient descent
    initial_stepsize: initial stepsize used in backtracking line search
    alpha: parameter in Armijo's rule
                f(x - t * f'(x)) > f(x) - t * alpha * ||f'(x)||^2
    beta: constant factor used in stepsize reduction
    tol: toleracne parameter in the stopping crieterion. Gradient descent stops
         when the 2-norm of the gradient is smaller than tol
    maxiter: maximum number of iterations in gradient descent.

    This function should return a list of the sequence of approximate solutions
    x_k produced by each iteration and the total number of iterations in the inner loop
    """
    x_traces = [x0]
    stepsize_traces = []

    x = x0
    #   START OF YOUR CODE
    while np.linalg.norm(fp(x)) > tol and len(x_traces) <= maxiter:
        d = -1 / fpp(x) * fp(x)
        stepsize = initial_stepsize
        while f(x + stepsize * d) > f(x) + alpha * stepsize * fp(x) * d:
            stepsize *= beta
        stepsize_traces.append(stepsize)

        x += stepsize * d
        x_traces.append(x)
    #   END OF YOUR CODE
    return x_traces, stepsize_traces