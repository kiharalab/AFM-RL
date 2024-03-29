# File: agent_brain.py
# Description: Creating brain for the agent based on the Q-learning
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Reinforcement Learning Algorithms for global path planning // GitHub platform. DOI: 10.5281/zenodo.1317899



# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# Importing function from the env.py
#from env import final_states


# Creating class for the Q-learning table
class QLearningTable:
    def __init__(self, actions,log,iterations,energy_details,learning_rate=0.1, reward_decay=1, e_greedy=1.0):
        # List of actions
        self.actions = actions
        self.energytable = energy_details
        # Learning rate
        self.lr = learning_rate
        self.init_lr = learning_rate
        self.drop_rate = 0.85
        self.episodes_drop = 1000
        self.decay_rate_list = []

        # Value of gamma
        self.gamma = 0.25#reward_decay
        # Value of epsilon
        self.epsilon = e_greedy
        self.eps_decay = 1.0 /iterations # slowly move from exploration to exploitation
        # Creating full Q-table for all cells
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # Creating Q-table for cells of the final route
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.logger = log
        self.logger.info("QTABLE PARAM : learning_rate : " + str(learning_rate) + "reward_decay : " + str(reward_decay) + "e_greedy : " + str(e_greedy))

    # Function for choosing the action for the agent
    def choose_action(self, observation):
        # Checking if the state exists in the table
        observation_str = str(observation)
        self.check_state_exist(observation_str)
        # Selection of the action - 90 % according to the epsilon == 0.9
        # Choosing the best action
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation_str, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # Choosing random action - left 10 % for choosing randomly
            if np.random.uniform() <= 0.5:
                action = np.random.choice(self.actions)
            else:
                if observation_str == 'None':
                    action = np.random.choice(self.actions)
                else:
                    #this indexing is because observation is string and i need to get the last chain that is being added
                    observation_index = str(observation[-1]) # observation[-7:-1]
                    action = np.int64(self.energytable[observation_index][0])
        return action

    def reset_qtable(self):
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def get_top5_actions(self,observation):
        if np.random.uniform() <= 0.5:
                observation_index = observation[-1] #observation[-7:-1]
                #print('observation: ', observation,' observation_index:',observation_index)#,'\t',self.energytable,'\t',observation)
                return list(self.energytable[str(observation_index)])
        else:
                self.check_state_exist(str(observation))
                state_action = self.q_table.loc[str(observation),:]
                action = list(state_action.nlargest().index)
                return action


    def get_top100_actions(self,observation):
        self.check_state_exist(observation)
        state_action = self.q_table.loc[observation,:]
        action = list(state_action.nlargest(100).index)
        return action


    # Function for learning and updating Q-table with new knowledge
    def learn(self, state, action, reward, next_state,episode):
        # Checking if the next step exists in the Q-table
        self.check_state_exist(next_state)

        # Current state in the current position
        q_predict = self.q_table.loc[state, action]
        current_val = q_predict

        # Checking if the next state is free or it is obstacle or goal
        if next_state != 'goal':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward

        # Updating Q-table with new knowledge
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

        if next_state == 'goal':
            if self.epsilon > 0:
                self.epsilon -= self.eps_decay
            if self.lr > 0:
               # self.lr -= self.eps_decay
               self.lr = self.step_decay(episode)

        #if next_state == 'goal':
            #new_val = self.q_table.loc[state, action]
            #print('current_val: ',current_val,' new_val: ',new_val, ' q_target: ',q_target)
        return self.q_table.loc[state, action]

    def step_decay(self,episode):
        lrate = self.init_lr * math.pow(self.drop_rate,math.floor((episode)/self.episodes_drop))
        self.decay_rate_list.append(lrate)
        return lrate

    def learn_backward(self, state, action, reward, next_state):
        # Checking if the next step exists in the Q-table
        self.check_state_exist(next_state)

        # Current state in the current position
        q_predict = self.q_table.loc[state, action]
        current_val = q_predict

        # Checking if the next state is free or it is obstacle or goal
        if next_state != 'goal':
            q_target = self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward

        # Updating Q-table with new knowledge
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

        if next_state == 'goal':
            if self.epsilon > 0:
                self.epsilon -= self.eps_decay

        return self.q_table.loc[state, action]

    def backpropagate_reward(self,transition_tracker,reward):
        for x in sorted(transition_tracker.keys(), reverse=True):
            state_data = transition_tracker[x]
            state = str(state_data[0])
            next_state = str(state_data[1])
            action = state_data[2]
            #goal state already updated previously, no need to update None state
            #if next_state == 'goal' or state == 'None':
            #    continue # this has been updated already with the initial learn() call
            #else:
            self.learn_backward(state,action,reward,next_state)
            #    #print('BACKPROPAGATING reward for state : ', state)

    # Adding to the Q-table new states
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            #print("Visiting new state....")
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


    def print_lr_decay(self):
        return np.unique(self.decay_rate_list)


    # Printing the Q-table with states
    def print_q_table(self,qtable_path):
        # Getting the coordinates of final route from env.py
        #e = final_states()

        # Comparing the indexes with coordinates and writing in the new Q-table values
        #for i in range(len(e)):
        #    state = str(e[i])  # state = '[5.0, 40.0]'
        #    # Going through all indexes and checking
        #    for j in range(len(self.q_table.index)):
        #        if self.q_table.index[j] == state:
        #            self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        #print()
        #print('Length of final Q-table =', len(self.q_table_final.index))
        #print('Final Q-table with values from the final route:')
        #print(self.q_table_final)

        # print()
        # print('Length of full Q-table =', len(self.q_table.index))
        # print('Full Q-table:')
        # print(self.q_table)
        #print(self.q_table.describe())
        self.q_table.to_csv(qtable_path)

    def print_intermediate_q_table(self,qtable_out,episode):
        path = qtable_out + str(self.lr) + '_' + str(episode) + 'qtable_snapshot.csv'
        self.q_table.to_csv(path)

    # Plotting the results for the number of steps
    def plot_results(self, steps, cost):
        #
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        #
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        #
        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        plt.tight_layout()  # Function to make distance between figures

        #
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        #
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, 'r')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        # Showing the plots
plt.show()
