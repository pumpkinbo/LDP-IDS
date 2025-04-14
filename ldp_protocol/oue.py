# -*- coding: utf-8 -*-
# @Time    : 2025-04-11 10:55
# @Author  : sun bo
# @File    : oue.py
# @Software: PyCharm
import numpy as np


def OUE(epsilon, d, u, user):
    """
    Frequency Oracle: OUE (Optimized Unary Encoding)

    Parameters:
        epsilon (float): Privacy budget
        d (int): Domain size
        u (np.ndarray): Original dataset (1D array of integers in [1, d])
        user (np.ndarray or list): Indices of sampled users (0-based indexing)

    Returns:
        np.ndarray: Estimated frequency count (after debiasing)
    """
    sum_value = np.zeros(d)  # Aggregated perturbed data
    p = 0.5
    q = 1 / (np.exp(epsilon) + 1)

    for j in user:
        value_j = np.zeros(d)
        Pvalue_j = np.zeros(d)

        value_j[u[j] - 1] = 1  # Set 1 at the true value position (adjusted to 0-based)

        for i in range(d):
            temp = np.random.rand()
            if value_j[i] == 1:
                if temp <= p:
                    Pvalue_j[i] = 1
            else:
                if temp <= q:
                    Pvalue_j[i] = 1

        sum_value += Pvalue_j

    estimated_count = (sum_value - len(user) * q) / (p - q)
    return estimated_count