import random as rnd
import matplotlib.pyplot as plt
import numpy as np

results = open("./2_5results.csv",'w')

#need to initialize q-values for the bandit, let's make them 0 centered
base_q_values = [0.0]

#set the test parameters
max_time_steps = 10000
max_runs = 2000

#parameters for random increment of bandits q vals
random_walk_mu = 0
random_walk_sig = 0.01

#the types of runs we will be comparing
run_types = ['constant_alpha','sample_average_alpha']

#model learning rate and exploration parameters
alpha = 0.1
epsilon = 0.1

#given the current estimate of q vals, select an action e-greedily
def select_action(q_estimate):
    greedy_action_index = q_estimate.argmax()

    if rnd.random()<epsilon:
        non_greedy_options = list(range(0,len(q_estimate)))
        del non_greedy_options[greedy_action_index]
        return rnd.choice(non_greedy_options)

    else:
        return greedy_action_index

#return the learning rate given the run type
def get_learning_rate(run_type, steps_count):
    if run_type=='constant_alpha':
        return 0.1
    elif run_type=='sample_average_alpha':
        return 1/(steps_count+1)

#randomly walk the q vals of the bandits
def adjust_q_values(q_values):

    for index, value in enumerate(q_values):
        walk_amount = rnd.gauss(random_walk_mu,random_walk_sig)
        q_values[index] += walk_amount

#train the agent given a base_q_value and run type
def train_agent(base_q_value,run_type):

    curr_step_count = 0
    curr_run_count = 0

    #per episode average of rewards and optimal action percentages
    curr_reward_avg_vector = np.zeros(max_time_steps,dtype='f')
    optimal_action_percentages = np.zeros(max_time_steps,dtype='f')

    #overall test average of rewards and optimal action percentages
    curr_reward_matrix = np.zeros([max_runs,max_time_steps])
    optimal_action_matrix = np.zeros([max_runs,max_time_steps])

    while curr_run_count<max_runs:
        q_estimate = np.zeros(10)
        #instantiate original q_values to value
        curr_q_values = np.full((10),base_q_value)

        #run through episode
        while curr_step_count<max_time_steps:
            #determine learning rate
            learning_rate = get_learning_rate(run_type,curr_step_count)

            #select action
            action = select_action(q_estimate)

            #check if action taken was optimal
            optimal_action = curr_q_values.argmax()
            optimal_chosen = 0

            if optimal_action==action:
                optimal_chosen = 1
            else:
                optimal_chosen = 0

            if curr_step_count==0:
                optimal_action_percentages[curr_step_count] = optimal_chosen
            else:
                optimal_action_percentages[curr_step_count] = (optimal_action_percentages[curr_step_count-1]*curr_step_count+optimal_chosen)/(curr_step_count+1)

            #add reward to reward averages
            reward = curr_q_values[action]

            if curr_step_count==0:
                curr_reward_avg_vector[curr_step_count] = reward
            else:
                curr_reward_avg = curr_reward_avg_vector[curr_step_count-1]
                new_reward_avg = (curr_reward_avg*(curr_step_count)+reward)/(curr_step_count+1)
                curr_reward_avg_vector[curr_step_count] = new_reward_avg

            #update the q val estimates towards the target
            q_estimate[action] = q_estimate[action] + learning_rate*(reward-q_estimate[action])

            #randomly walk q_values
            adjust_q_values(curr_q_values)

            curr_step_count += 1

        #add results of that episode to the overall results matrix
        curr_reward_matrix[curr_run_count] = curr_reward_avg_vector
        optimal_action_matrix[curr_run_count] = optimal_action_percentages

        curr_run_count += 1

    #return the average results per run
    return [np.mean(curr_reward_matrix,axis=0),np.mean(optimal_action_matrix,axis=0)]

#plot the results
x_range_list = list(range(1,max_time_steps+1))

figure = plt.figure(1)

reward_subplot = figure.add_subplot(2,1,1)
optimal_action_subplot = figure.add_subplot(2,1,2)

for base_q_value in base_q_values:

    for run_type in run_types:
        result_list = train_agent(base_q_value,run_type)

        reward_vector = result_list[0]
        optimal_action_vector = result_list[1]

        reward_subplot.plot(x_range_list,reward_vector)
        optimal_action_subplot.plot(x_range_list,optimal_action_vector)

        #write results to csv, in case needed later
        results.write((','.join(map(str,reward_vector.tolist())))+'\n')
        results.write((','.join(map(str,reward_vector.tolist())))+'\n')

reward_subplot.set_title('Average rewards over {0} runs'.format(max_runs))
reward_subplot.legend(run_types)
reward_subplot.set_xlabel('Time steps')
reward_subplot.set_ylabel('Average reward')

optimal_action_subplot.set_title('Avg optimal action percentage over {0} runs'.format(max_runs))
optimal_action_subplot.legend(run_types)
optimal_action_subplot.set_xlabel('Time steps')
optimal_action_subplot.set_ylabel('Average % of optimal action')

figure.tight_layout()
plt.show()

results.close()
