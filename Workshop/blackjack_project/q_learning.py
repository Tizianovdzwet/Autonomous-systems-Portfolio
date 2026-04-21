#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:02:13 2020
@author: joshuageiser
"""
import os
import pandas as pd
import numpy as np
import time

class const():
    def __init__(self):
        self.gamma = 0.5
        self.input_filename = 'random_policy_runs_mapping_2.csv'
        self.output_filename = 'QLearning_policy_mapping_2.policy'
        self.n_states = 183  # 21 for state mapping 1, 183 for state mapping 2
        self.n_action = 2
        self.alpha = 0.01
        self.lambda_ = 0.1

def update_q_learning(Q_sa, df_i, CONST):
    """Perform Q-Learning update for a single sample."""
    diff = df_i.r + (CONST.gamma * max(Q_sa[df_i.sp])) - Q_sa[df_i.s][df_i.a]
    Q_sa[df_i.s][df_i.a] += CONST.alpha * diff
    return


def train_q(input_file, CONST):
    """Train a policy using Q-learning."""
    df = pd.read_csv(input_file)

    Q_sa = np.zeros((CONST.n_states, CONST.n_action))

    for i in range(len(df)):
        df_i = df.loc[i]
        update_q_learning(Q_sa, df_i, CONST)

    policy = np.argmax(Q_sa, axis=1)
    write_outfile(policy, CONST)
    return


def write_outfile(policy, CONST):
    """Write policy to a .policy output file."""
    output_dir = os.getcwd()
    output_file = os.path.join(output_dir, f'{CONST.output_filename}')

    df = open(output_file, 'w')
    for i in range(CONST.n_states):
        df.write(f'{policy[i]}\n')
    df.close()

    print(f"Policy saved to {output_file}")
    return


def main():
    start = time.time()
    CONST = const()
    input_file = os.path.join(os.getcwd(), CONST.input_filename)
    train_q(input_file, CONST)

    end = time.time()
    print(f'Total time: {end-start:0.2f} seconds')
    print(f'Total time: {(end-start)/60:0.2f} minutes')
    return


if __name__ == '__main__':
    main()
