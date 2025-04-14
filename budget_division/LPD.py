# -*- coding: utf-8 -*-
# @Time    : 2025-04-14 15:31
# @Author  : sun bo
# @File    : LPD.py
# @Software: PyCharm
import numpy as np
import math
from ldp_protocol.oue import OUE
from ldp_protocol.grr import GRR
from Var import Var

def LPD(u, d, w, e, rounds, c, beta):
    """
    LPD algorithm for differentially private frequency estimation (population distribution strategy).

    Parameters:
        u (np.ndarray): Dataset of shape (N, T), each entry in [1, d]
        d (int): Domain size
        w (int): Length of time window
        e (float): Privacy budget
        rounds (int): Number of runs
        c (np.ndarray): Real frequency stream, shape (d, T)
        beta (float): Ratio of privacy budget allocated to M_{t,1}

    Returns:
        np.ndarray: [MRE, MAE, MSE, communication per user]
    """
    N, T = u.shape
    LPD_error = np.zeros(4)
    MAE_LPD = np.zeros(rounds)
    MRE_LPD = np.zeros(rounds)
    MSE_LPD = np.zeros(rounds)
    Communication_users = np.zeros(rounds)

    for z in range(rounds):
        user_number = np.zeros(T)
        cbd1 = np.zeros((d, T))
        cbd2 = np.zeros((d, T))
        cbd = np.zeros((d, T))
        AE = np.zeros(T)
        RE = np.zeros(T)
        SE = np.zeros(T)
        Bits = np.zeros(T)

        remain_size = np.zeros(T)
        dis = np.zeros(T)

        N1 = math.floor(beta * N / w)
        user_index = np.random.permutation(N)
        user_available = list(user_index)
        user_used1 = [set() for _ in range(T)]
        user_used2 = [set() for _ in range(T)]

        for t in range(T):
            user_sample1 = np.random.choice(user_available, N1, replace=False).tolist()
            user_used1[t] = set(user_sample1)
            user_available = list(set(user_available) - set(user_sample1))

            if d <= (3 * np.exp(e) + 2):
                cbd1[:, t] = GRR(e, d, u[:, t], user_sample1) * (N / len(user_sample1))
                Bits[t] += len(user_sample1) * (math.ceil(math.log2(d)) + 1)
            else:
                cbd1[:, t] = OUE(e, d, u[:, t], user_sample1) * (N / len(user_sample1))
                Bits[t] += len(user_sample1) * (d + 1)

            var1, var_extra1 = Var(d, e, N, N1)

            if t == 0:
                Npp = math.floor((N - w * N1) / 2)
                flag = True
            else:
                dis[t] = np.sum((cbd1[:, t] - cbd[:, ft]) ** 2) / d - (var1 + var_extra1)

                used_size = 0
                for k in range(1, w + 1):
                    if t >= k - 1:
                        used_size += len(user_used2[t - k + 1])
                remain_size[t] = N - w * N1 - used_size
                Npp = math.floor(remain_size[t] / 2)

                if Npp <= 2:
                    flag = False
                else:
                    var2, var_extra2 = Var(d, e, N, Npp + N1)   # why not be Npp?
                    err = var2 + var_extra2
                    flag = dis[t] > err

            if flag:
                ft = t
                user_sample2 = np.random.choice(user_available, Npp, replace=False).tolist()
                user_used2[t] = set(user_sample2)
                user_available = list(set(user_available) - set(user_sample2))

                if d <= (3 * np.exp(e) + 2):
                    cbd2[:, t] = GRR(e, d, u[:, t], user_sample2) * (N / len(user_sample2))
                    Bits[t] += len(user_sample2) * (math.ceil(math.log2(d)) + 1)
                else:
                    cbd2[:, t] = OUE(e, d, u[:, t], user_sample2) * (N / len(user_sample2))
                    Bits[t] += len(user_sample2) * (d + 1)

                N_total = N1 + len(user_sample2)
                cbd[:, t] = (N1 * cbd1[:, t] + len(user_sample2) * cbd2[:, t]) / N_total
                user_number[t] = N_total
            else:
                cbd[:, t] = cbd[:, t - 1]
                user_number[t] = N1

            if t >= w - 1:  # 0-based index
                user_available = list(set(user_available)
                                      | user_used1[t - w + 1]
                                      | user_used2[t - w + 1])

            c_max = np.maximum(c[:, t], 1)
            AE[t] = np.sum(np.abs(cbd[:, t] - c[:, t])) / d
            RE[t] = np.sum(np.abs(cbd[:, t] - c[:, t]) / c_max) / d
            SE[t] = np.sum((cbd[:, t] - c[:, t]) ** 2) / d

        MAE_LPD[z] = np.mean(AE)
        MRE_LPD[z] = np.mean(RE)
        MSE_LPD[z] = np.mean(SE)
        Communication_users[z] = np.mean(Bits) / N

    LPD_error[0] = np.mean(MRE_LPD)
    LPD_error[1] = np.mean(MAE_LPD)
    LPD_error[2] = np.mean(MSE_LPD)
    LPD_error[3] = np.mean(Communication_users)

    return LPD_error
