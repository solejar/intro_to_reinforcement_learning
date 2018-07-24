import random as rnd
import matplotlib.pyplot as plt
import numpy as np

results = open("./2_5results.csv",w)

#need to initialize q-values for the bandit
base_q_values = [-2,0,2]

max_time_steps = 10000
max_runs = 2000
#reward_array = np.zeros([len(run_types),len(base_q_values),max_time_steps])

random_walk_mu = 0
random_walk_sig = 0.01

run_types = ['constant_alpha','sample_average_alpha']

alpha = 0.1

def get_learning_rate(run_type, steps_count):
    if run_type=='constant_alpha':
        return 0.1
    elif run_type=='sample_average_alpha':
        return 1/steps_count

def train_agent(base_q_value,run_type):

    curr_step_count = 0
    curr_run_count = 0

    while curr_run_count<max_runs:

        while curr_step_count<max_time_steps:
            learning_rate = get_learning_rate(run_type,curr_step_count)
            curr_step_count += 1
        curr_run_count += 1

x_range_list = list(range(1,max_time_steps+1))

for x_index, base_q_value in enumerate(base_q_values):

    for y_index,run_type in enumerate(run_types):
        reward_vector = train_agent(base_q_value,run_type)

        plt.plot(x_range_list,reward_vector)

        results.write(''.join(reward_vector)+'\n')

results.close()
