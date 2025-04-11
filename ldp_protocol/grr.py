# -*- coding: utf-8 -*-
# @Time    : 2025-04-11 10:07
# @Author  : sun bo
# @File    : grr.py
# @Software: PyCharm

import numpy as np


def GRR(epsilon, d, u, user):
    """
    Frequency Oracle: GRR (Generalized Randomized Response)

    Parameters:
        epsilon (float): Privacy budget
        d (int): Domain size
        u (np.ndarray): Original dataset (1D array of integers in [1, d])
        user (np.ndarray or list): Indices of sampled users (0-based indexing)

    Returns:
        np.ndarray: Estimated frequency count (after debiasing)
    """
    p = np.exp(epsilon) / (d - 1 + np.exp(epsilon))
    q = 1 / (d - 1 + np.exp(epsilon))

    up = np.zeros_like(u)  # Perturbed data
    sumc = np.zeros(d)  # Frequency of perturbed data

    for j in user:
        temp = np.random.rand()
        if temp <= p:
            up[j] = u[j]
        else:
            sign = u[j]
            sign_perturb = int(np.ceil(np.random.rand() * (d - 1))) + sign
            if sign_perturb > d:
                un = sign_perturb % d
                if un == 0:
                    un = d
            else:
                un = sign_perturb
            up[j] = un
        sumc[int(up[j]) - 1] += 1  # Convert to 0-based index

    estimated_freq = (sumc - len(user) * q) / (p - q)
    return estimated_freq  # count
