# -*- coding: utf-8 -*-
# @Time    : 2025-04-14 8:52
# @Author  : sun bo
# @File    : Var.py
# @Software: PyCharm
import numpy as np

def Var(d, e, N, Ns):
    """
    Compute the variance and extra variance of count estimation
    based on the domain size and privacy budget.

    Parameters:
        d (int): Domain size
        e (float): Privacy budget
        N (int): Number of users
        Ns (int): Sampled user count (usually same as N)

    Returns:
        tuple: (var, var_extra)
    """
    if d < (3 * np.exp(e) + 2):
        # Variance for GRR (variance of count rather than frequency)
        var = (N * N / Ns) * ((d - 2) + np.exp(e)) / ((np.exp(e) - 1) ** 2)
        var_extra = (1 / d) * (N * N / Ns) * (d - 2) / (np.exp(e) - 1)
    else:
        # Variance for OUE
        var = (N * N / Ns) * (4 * np.exp(e)) / ((np.exp(e) - 1) ** 2)
        var_extra = (1 / d) * (N * N / Ns)

    return var, var_extra
