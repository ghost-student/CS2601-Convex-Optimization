import numpy as np


def newton(fp, fpp, x0, A, b, tol=1e-5, maxiter=100000):
    """
    fp: function that takes an input x and returns the gradient of f at x
    fpp: function that takes an input x and returns the Hessian of f at x
    x0: initial point
    tol: toleracne parameter in the stopping crieterion. Newton's method stops 
         when the 2-norm of the gradient is smaller than tol
    maxiter: maximum number of iterations

    This function should return a list of iterates x_k produced by each iteration
    """
    x_traces = [np.array(x0)]
    x = np.array(x0)
    d = np.array([float('inf')])
    #   START OF YOUR CODE
    while np.linalg.norm(d) >= tol and len(x_traces) <= maxiter:
        fpp_inv = np.linalg.inv(fpp(x))
        d = fpp_inv @ (A.T @ np.linalg.inv(A @ fpp_inv @ A.T) @ A @ fpp_inv @ fp(x) - fp(x))
        x += d
        x_traces.append(np.array(x))
    #   END OF YOUR CODE
    return x_traces


def newton_eq(f, fp, fpp, x0, A, b, initial_stepsize=1.0,
              alpha=0.5, beta=0.5, tol=1e-5, maxiter=100000):
    """
    f: function that takes an input x an returns the value of f at x
    fp: function that takes an input x and returns the gradient of f at x
    fpp: function that takes an input x and returns the Hessian of f at x
    A, b: constraint A x = b
    x0: initial feasible point
    initial_stepsize: initial stepsize used in backtracking line search
    alpha: parameter in Armijo's rule 
                f(x + t * d) > f(x) + t * alpha * f(x) @ d
    beta: constant factor used in stepsize reduction
    tol: toleracne parameter in the stopping crieterion. Gradient descent stops 
         when the 2-norm of the Newton direction is smaller than tol
    maxiter: maximum number of iterations in outer loop of damped Newton's method.

    This function should return a list of the iterates x_k, a list of setpsizes
    """
    x_traces = [np.array(x0)]
    stepsize_traces = []

    x = np.array(x0)
    d = np.array([float('inf')])
    #   START OF YOUR CODE
    while np.linalg.norm(d) >= tol:
        fpp_inv = np.linalg.inv(fpp(x))
        d = fpp_inv @ (A.T @ np.linalg.inv(A @ fpp_inv @ A.T) @ A @ fpp_inv @ fp(x) - fp(x))

        stepsize = initial_stepsize
        while f(x + stepsize * d) > f(x) + alpha * stepsize * fp(x).T @ d:
            stepsize *= beta
        stepsize_traces.append(stepsize)

        x += stepsize * d
        x_traces.append(np.array(x))
    #   END OF YOUR CODE
    return x_traces, stepsize_traces
