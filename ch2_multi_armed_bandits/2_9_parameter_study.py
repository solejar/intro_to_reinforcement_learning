import random as rnd
import matplotlib.pyplot as plt
import numpy as np

def train_agent(run_type,parameter_ranges,total_steps,threshold_to_count_results):

    curr_step = 0
    results = []

    for parameter in paramter_range:
        result = 0.0
        while curr_step<total_steps:

            action = select_action(run_type)
            reward = q_values[action]

            if curr_step>=threshold_to_count_results:
                result = (result*curr_step + reward)/(curr_step+1)

            curr_step += 1
        results.append(result)

    return results


def plot(run_types,parameter_ranges,results):

    plt.figure(1)

    for run_type in run_types:
        plt.log(parameter_ranges[run_type],results[run_type])

    plt.title('Param study for 4 multi-armed bandit algos trained over {0} steps'.format(total_steps))
    plt.legend(run_types)
    plt.xlabel(u'\u03B5',u'\u03B1','c','Q'+u'\u2080')
    plt.ylabel('Average reward of last {0} steps'.format(threshold_to_count_results))


if __name__ == "__main__":

    total_steps = 200000
    threshold_to_count_results = 100000

    run_types = [
        'constant_step_e_greedy',
        'gradient_bandit',
        'upper_confidence_bound',
        'greedy_optimistic_initialization'
    ]

    parameter_ranges = {
        'constant_step_e_greedy': [1/128,1/64,1/32,1/16,1/4],
        'gradient_bandit': [1/32,1/16,1/8,1/4,1/2,1,2],
        'upper_confidence_bound': [1/16,1/8,1/4,1/2,1,2,4],
        'greedy_optimistic_initialization': [1/4,1/2,1,2,4]
    }

    results = {}

    for run_type in run_types:
        results[run_type] = train_agent(run_type,parameter_ranges,total_steps,threshold_to_count_results)

    plot(run_types,parameter_ranges,results)
