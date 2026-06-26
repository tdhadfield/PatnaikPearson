import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import qr

import math
import pandas as pd

import random

import PatnaikPearson as pp
import cupy as cp

import torch
use_gpu = torch.cuda.is_available()

num_iterations = 10
d = 10
alpha_A = 5.0
alpha_B = 5.0

uniform_draws = False
use_pareto = True
use_uniform = False
use_cauchy = False
verbose = False

these_lower_bound_over_actual_vals = np.zeros(num_iterations)
these_upper_bound_over_actual_vals = np.zeros(num_iterations)

for i in range(num_iterations):
    A = pp.generate_square_weight_matrix(d, alpha_A, uniform_draws, use_pareto, use_uniform, use_cauchy, verbose)
    pp_dim_A = pp.calculate_PatnaikPearson_dim(A)
    dim_A = A.shape[1]
    nu_over_d_A = pp_dim_A / dim_A
    B = pp.generate_square_weight_matrix(d, alpha_A, uniform_draws, use_pareto, use_uniform, use_cauchy, verbose)
    pp_dim_B = pp.calculate_PatnaikPearson_dim(B)
    dim_B = B.shape[1]
    nu_over_d_B = pp_dim_B / dim_B

    if use_gpu:
        AB = cp.asnumpy(cp.matmul(cp.array(A), cp.array(B)))
    else:
        AB = A @ B

    pp_dim_AB = pp.calculate_PatnaikPearson_dim(AB)
    dim_AB = AB.shape[1]
    nu_over_d_AB = pp_dim_AB / dim_AB

    lower_bound = nu_over_d_A * nu_over_d_B
    lower_bound_over_actual = lower_bound / nu_over_d_AB
    upper_bound = min(nu_over_d_A, nu_over_d_B)
    upper_bound_over_actual = upper_bound / nu_over_d_AB

    these_lower_bound_over_actual_vals[i] = lower_bound_over_actual
    these_upper_bound_over_actual_vals[i] = upper_bound_over_actual

    print(i, lower_bound_over_actual, upper_bound_over_actual)

df = pd.DataFrame(data=[these_lower_bound_over_actual_vals, these_upper_bound_over_actual_vals]).T
df.columns = ["lower bound / actual", "upper bound / actual"]
df.head(5)
file_name = "tmp_20260626.csv"
df.to_csv(file_name)