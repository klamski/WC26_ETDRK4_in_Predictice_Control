import math
import scipy.linalg
import numpy as np


def phi_n_scalar(z, n, terms=1):
    if abs(z) < 1e-6:
        return sum(z**k / math.factorial(k + n) for k in range(terms))
    else:
        if n == 1:
            return (np.exp(z) - 1) / z
        elif n == 2:
            return (np.exp(z) - 1 - z) / (z**2)
        elif n == 3:
            return (np.exp(z) - 1 - z - z**2 / 2) / (z**3)


def phi_n_matrix(L, n, terms=1):
    eigvals, V = scipy.linalg.eigh(L)
    V_inv = scipy.linalg.inv(V)
    phi_diag = np.diag([phi_n_scalar(lam, n, terms) for lam in eigvals])
    return V @ phi_diag @ V_inv
    


def etdrk4(A, f, x0, dt):
    E = scipy.linalg.expm(A * dt)
    E2 = scipy.linalg.expm(A * dt / 2)

    Q = dt / 2 * phi_n_matrix(A * dt / 2, 1)

    phi1 = phi_n_matrix(A * dt, 1)
    phi2 = phi_n_matrix(A * dt, 2)
    phi3 = phi_n_matrix(A * dt, 3)

    f0 = f(x0)
    a = E2 @ x0 + Q @ f0
    fa = f(a)
    b = E2 @ x0 + Q @ fa
    fb = f(b)
    c = E2 @ a + Q @ (2 * fb - f0)
    fc = f(c)

    k1 = phi1 - 3 * phi2 + 4 * phi3
    k2 = 2 * phi2 - 4 * phi3
    k3 = -1 * phi2 + 4 * phi3

    x = E @ x0 + dt * (k1 @ f0 + k2 @ (fa + fb) + k3 @ fc)

    return x


def etdrk2(A, f, x0, dt):
    phi1 = phi_n_matrix(A * dt, 1)
    phi2 = phi_n_matrix(A * dt, 2)
    a = f(x0)
    b = scipy.linalg.expm(A * dt) @ x0 + dt * phi1 @ a
    x = b + dt * phi2 @ (f(b) - a)

    return x

def etdrk1(A, f, x0, dt):
    phi1 = phi_n_matrix(A * dt, 1)
    a = f(x0)
    b = scipy.linalg.expm(A * dt) @ x0 + dt * phi1 @ a
    x = b 

    return x
