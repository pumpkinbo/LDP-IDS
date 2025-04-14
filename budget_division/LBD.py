# -*- coding: utf-8 -*-
# @Time    : 2025-04-11 11:47
# @Author  : sun bo
# @File    : LBD.py
# @Software: PyCharm
import numpy as np
import math
from ldp_protocol import oue

def LBD(u, d, w, e, rounds, c, beta):
    """
    LBD algorithm for differentially private frequency estimation.

    Parameters:
        u (np.ndarray): Dataset of shape (N, T), each entry in [1, d]
        d (int): Domain size
        w (int): Length of the time window
        e (float): Total privacy budget
        rounds (int): Number of repeated runs
        c (np.ndarray): Real frequency stream, shape (d, T)
        beta (float): Ratio of privacy budget allocated to M_{t,1}

    Returns:
        np.ndarray: Mean MRE, MAE, MSE, communication per user
    """
    N, T = u.shape
    LBD_error = np.zeros(4)
    MAE_LBD = np.zeros(rounds)
    MRE_LBD = np.zeros(rounds)
    MSE_LBD = np.zeros(rounds)
    Communication_user = np.zeros(rounds)

    user = np.arange(N)  # index set for users, 0-based

    for z in range(rounds):
        Bits = np.zeros(T)
        e1 = beta * e / w
        e2 = np.zeros(T)

        cbd1 = np.zeros((d, T))
        cbd2 = np.zeros((d, T))
        cbd = np.zeros((d, T))

        AE = np.zeros(T)
        RE = np.zeros(T)
        SE = np.zeros(T)

        for t in range(T):
            # M_{t,1}
            if d <= (3 * np.exp(e1) + 2):
                cbd1[:, t] = GRR(e1, d, u[:, t], user)
                Bits[t] = N * math.ceil(math.log2(d))
            else:
                cbd1[:, t] = OUE(e1, d, u[:, t], user)
                Bits[t] = N * d

            var1, var_extra1 = Var(d, e1, N, N)

            if t == 0:
                e2[t] = (1 - beta) * e / 2
                ft = t
                if d <= (3 * np.exp(e2[t]) + 2):
                    cbd2[:, t] = GRR(e2[t], d, u[:, t], user)
                    Bits[t] += N * (math.ceil(math.log2(d)) + 1)
                else:
                    cbd2[:, t] = OUE(e2[t], d, u[:, t], user)
                    Bits[t] += N * (d + 1)
                cbd[:, t] = cbd2[:, t]
            else:
                # dis computation
                dis = np.sum((cbd1[:, t] - cbd[:, ft]) ** 2) / d - (var1 + var_extra1)

                used_eps = 0
                for k in range(1, w + 1):
                    if t >= k:
                        used_eps += e2[t - k + 1]
                e2[t] = ((1 - beta) * e - used_eps) / 2

                var2, var_extra2 = Var(d, e2[t], N, N)
                err = var2 + var_extra2
                flag = dis > err

                if flag:
                    ft = t
                    if d <= (3 * np.exp(e2[t]) + 2):
                        cbd2[:, t] = GRR(e2[t], d, u[:, t], user)
                        Bits[t] += N * (math.ceil(math.log2(d)) + 1)
                    else:
                        cbd2[:, t] = OUE(e2[t], d, u[:, t], user)
                        Bits[t] += N * (d + 1)
                    cbd[:, t] = cbd2[:, t]
                else:
                    e2[t] = 0
                    cbd[:, t] = cbd[:, t - 1]

            # compute errors
            c_max = np.maximum(c[:, t], 1)
            AE[t] = np.sum(np.abs(cbd[:, t] - c[:, t])) / d
            RE[t] = np.sum(np.abs(cbd[:, t] - c[:, t]) / c_max) / d
            SE[t] = np.sum((cbd[:, t] - c[:, t]) ** 2) / d

        MAE_LBD[z] = np.mean(AE)
        MRE_LBD[z] = np.mean(RE)
        MSE_LBD[z] = np.mean(SE)
        Communication_user[z] = np.mean(Bits) / N

    LBD_error[0] = np.mean(MRE_LBD)
    LBD_error[1] = np.mean(MAE_LBD)
    LBD_error[2] = np.mean(MSE_LBD)
    LBD_error[3] = np.mean(Communication_user)

    return LBD_error
