import numpy as np
import random


def calculate_source_terms(L, E, K, x, t, A, B, m, n, w, z):
    U = (E * A * np.pi * np.exp(-t * m + n) + B * np.exp(-t * w + z)) * np.cos(np.pi * x / (2 * L)) * np.pi
    P = (K * B * np.pi * np.exp(-t * w + z) + A * m * np.exp(-t * m + n)) * np.sin(np.pi * x / (2 * L)) * np.pi
    return U, P


def calculate_initial_condition(L, x, A, B, n, z):
    u0 = A * np.cos(np.pi * x / (2 * L)) * np.exp(n)
    p0 = B * np.sin(np.pi * x / (2 * L)) * np.exp(z)
    return u0, p0


def calculate_real_solution(L, x, t, A, B, m, n, w, z):
    u = A * np.cos(np.pi * x / (2 * L)) * np.exp(-t * m + n)
    p = B * np.sin(np.pi * x / (2 * L)) * np.exp(-t * w + z)
    return u, p


def create_sample(L, t_f, E, K, size_x, size_t, batch_size, x, t):

    U_batch = []
    P_batch = []
    u0_batch = []
    p0_batch = []

    x_random = []
    t_random = []
    U_random = []
    P_random = []
    u0_random = []
    p0_random = []

    u_real = []
    p_real = []

    for _ in range(batch_size):

        A = random.uniform(0.5, 3)
        m = random.uniform(0.5, 2)
        n = random.uniform(-2, 0)

        B = random.uniform(0.5, 3)
        w = random.uniform(0.5, 2)
        z = random.uniform(-2, 0)
    
        U, P = calculate_source_terms(L, E, K, x, t, A, B, m, n, w, z)
        u0, p0 = calculate_initial_condition(L, x[0], A, B, n, z)

        U_batch.append(U.reshape((size_x * size_t)))
        P_batch.append(P.reshape((size_x * size_t)))
        u0_batch.append(u0)
        p0_batch.append(p0)

        x_random.append([random.uniform(0, L)])
        t_random.append([random.uniform(0, t_f)])

        U, P = calculate_source_terms(L, E, K, x_random[-1][0], 1, A, B, m, n, w, z)
        u0, p0 = calculate_initial_condition(L, x_random[-1][0], A, B, n, z)
        u, p = calculate_real_solution(L, x_random[-1][0], 1, A, B, m, n, w, z)

        U_random.append([U])
        P_random.append([P])

        u0_random.append([u0])
        p0_random.append([p0])

        u_real.append([u])
        p_real.append([p])

    batch_vars = [np.array(U_batch), np.array(P_batch), np.array(u0_batch), np.array(p0_batch)]
    random_vars = [np.array(x_random), np.array(t_random), np.array(U_random), np.array(P_random), np.array(u0_random), np.array(p0_random)]
    real_solution = [np.array(u_real), np.array(p_real)]

    return batch_vars, random_vars, real_solution
