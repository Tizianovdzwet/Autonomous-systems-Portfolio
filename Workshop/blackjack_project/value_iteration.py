#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 10:52:50 2020
@author: tristan
"""
import os
import pandas as pd
import numpy as np

#~~~~~~~~~~~~~~~~~
state_mapping = 2
#~~~~~~~~~~~~~~~~~

if state_mapping == 1:
    S = 21
    A = 2
    gam = 0.5
    maxIters = 100
    input_file = "random_policy_runs_mapping_1.csv"
    output_file = 'Value_Iteration_Policy_1.policy'
elif state_mapping == 2:
    S = 183
    A = 2
    gam = 0.5
    maxIters = 100
    input_file = "random_policy_runs_mapping_2.csv"
    output_file = 'Value_Iteration_Policy_2.policy'

input_file = os.path.join(os.path.dirname(__file__), input_file)
df = pd.read_csv(input_file)
s_data = df['s']
a_data = df['a']
r_data = df['r']
sp_data = df['sp']

U = np.zeros((S))
u = np.zeros((A))
T = np.zeros((S, S, A))
R = np.zeros((S, A))
rho = np.zeros((S, A))
N = np.zeros((S, A, S))
policy = np.zeros((S))

# Learn the model
for k in range(len(df)):
    s = s_data[k]
    a = a_data[k]
    r = r_data[k]
    sp = sp_data[k]

    N[s, a, sp] += 1
    rho[s, a] += r

    if(sum(N[s, a, :]) == 0):
        T[sp, s, a] = 0
        R[s, a] = 0
    else:
        T[sp, s, a] = N[s, a, sp] / sum(N[s, a, :])
        R[s, a] = rho[s, a] / sum(N[s, a, :])

# Value Iteration
for s in range(S):
    for i in range(maxIters):
        for a in range(A):
            u[a] = R[s,a] + gam*sum(T[:, s, a] * U[:])
        U[s] = max(u)
        if i == maxIters-1:
            policy[s] = np.argmax(u)

output_dir = os.getcwd()
output_file = os.path.join(output_dir, output_file)

DF = open(output_file, 'w')
for i in range(S):
    DF.write(f'{int(policy[i])}\n')
DF.close()

print(f"Policy saved to {output_file}")
