import random as rnd
import matplotlib.pyplot as plt
import numpy as np

def walk_q_values(q_values, random_walk_mu=0.0,random_walk_sig=0.01):
    for index, value in enumerate(q_values):
        walk_amount = rnd.gauss(random_walk_mu, random_walk_sig)
        q_values[index] += walk_amount

def select_action_e_greedy(q_estimates, epsilon):

    greedy_action_index = q_estimates.argmax()

    if rnd.random()<epsilon:
        non_greedy_options = list(range(0,len(q_estimates)))
        del non_greedy_options[greedy_action_index]
        return rnd.choice(non_greedy_options)
    else:
        return greedy_action_index

def select_action_gradient_bandit(action_preferences):

    #softmax over the action_preferences
    action_probabilities = np.exp(action_preferences)/np.sum(np.exp(action_preferences),axis=0)

    return action_probabilities.argmax()

def select_action_ucb(q_estimates, exploration_constant, action_select_count, curr_time_step):

    potential_uncertainty = exploration_constant*np.sqrt(np.log(curr_time_step+1)/(action_select_count + 0.0001)) #adding a small number to action count to simplify div-by-0 logic
    ucb_augmented_q_vals = q_estimates + potential_uncertainty

    return ucb_augmented_q_vals.argmax()

def select_action_greedy(q_estimates):

    return q_estimates.argmax()

def update_estimates(q_estimates,action, reward,learning_rate=0.1):
    q_estimates[action] = q_estimates[action]+learning_rate*(reward-q_estimates[action])

def update_preferences(action_preferences, action, reward, reward_baseline, learning_rate):

    #softmax over the action_preferences
    action_probabilities = np.exp(action_preferences)/np.sum(np.exp(action_preferences),axis=0)

    non_selected_actions = list(range(0,len(action_preferences)))
    del non_selected_actions[action]

    #update preference of selected_action
    action_preferences[action] = action_preferences[action] + learning_rate*(reward-reward_baseline)*(1-action_probabilities[action])

    #update preference of non-selected actions
    for non_selected_action in non_selected_actions:
        action_preferences[non_selected_action] = action_preferences[non_selected_action] - learning_rate*(reward-reward_baseline)*action_probabilities[action]

def train_agent(run_type,parameter_range,total_steps,threshold_to_count_results):

    curr_step = 0
    results = []

    for parameter in parameter_range:
        result = 0.0
        reward_baseline_gradient_bandit = 0.0

        initial_q_value = 0.0
        q_values = np.full((10),initial_q_value)

        initial_selection_values = 0.0
        if run_type=='greedy_optimistic_initialization':
            initial_selection_values = parameter

        selection_values = np.full((10),initial_selection_values)

        #count for number of times action has been selected
        action_select_count = np.full((10),0)

        while curr_step<total_steps:

            action = -1 #if run_type not accounted for, will be inaccurate
            if run_type == 'constant_step_e_greedy':
                action = select_action_e_greedy(selection_values, parameter)

            elif run_type == 'gradient_bandit':
                action = select_action_gradient_bandit(selection_values)

            elif run_type == 'upper_confidence_bound':
                action = select_action_ucb(selection_values, parameter, action_select_count, curr_step)
                action_select_count[action] += 1

            elif run_type == 'greedy_optimistic_initialization':
                action = select_action_greedy(selection_values)

            reward = q_values[action]

            if curr_step>=threshold_to_count_results:
                result = (result*curr_step + reward)/(curr_step+1)
            reward_baseline_gradient_bandit = (reward_baseline_gradient_bandit*curr_step + reward)/(curr_step+1)

            walk_q_values(q_values)

            if run_type == 'gradient_bandit':
                update_preferences(selection_values,action, reward, reward_baseline_gradient_bandit, parameter)

            elif run_type=='constant_step_e_greedy' or run_type=='greedy_optimistic_initialization' or run_type=='upper_confidence_bound':
                update_estimates(selection_values, action, reward)

            curr_step += 1

        results.append(result)

    return results

def plot(run_types,parameter_ranges,results):

    fig, ax = plt.subplots()
    ax.set_xscale('log',basex=2)

    for run_type in run_types:
        ax.plot(parameter_ranges[run_type],results[run_type])

    plt.title('Param study for 4 multi-armed bandit algos trained over {0} steps'.format(total_steps))
    plt.legend(run_types)
    plt.xlabel(u'\u03B5'+','+u'\u03B1'+',c,Q'+u'\u2080')
    plt.ylabel('Average reward of last {0} steps'.format(threshold_to_count_results))

    plt.show()

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
        #varying eps
        'constant_step_e_greedy': [1/128,1/64,1/32,1/16,1/4],
        #varying alpha
        'gradient_bandit': [1/32,1/16,1/8,1/4,1/2,1,2],
        #varying c
        'upper_confidence_bound': [1/16,1/8,1/4,1/2,1,2,4],
        #varying initial q vals
        'greedy_optimistic_initialization': [1/4,1/2,1,2,4]
    }

    results = {}

    for run_type in run_types:
        print('collecting results for: ' +run_type)
        results[run_type] = train_agent(run_type,parameter_ranges[run_type],total_steps,threshold_to_count_results)

    plot(run_types,parameter_ranges,results)
