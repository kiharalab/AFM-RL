# File: run_agent.py
# Description: Running algorithm
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Reinforcement Learning Algorithms for global path planning // GitHub platform. DOI: 10.5281/zenodo.1317899



# Importing classes
from env import Environment
from qtable import QLearningTable
from argparser import argparser
import logging
import time
import sys
import pandas as pd

def update(no_of_episodes,qtable_path):
    # Resulted list for the plotting Episodes via Steps
    steps = []

    # Summed costs for all episodes in resulted list
    all_costs = []

    for episode in range(no_of_episodes):
        # Initial Observation
        #print("Currently running episode:",episode)
        observation = env.reset() # root observation getroot()
        transition_tracker = {}
        # Updating number of Steps for each Episode
        i = 0

        # Updating the cost for each episode
        cost = 0

        while True:
            # Refreshing environment
            #env.render() # no need

            # RL chooses action based on observation
            action = RL.choose_action(str(observation)) # selects best decoy or random decoy base on chance

            # RL takes an action and get the next observation and reward
            observation_, reward, done = env.step(action,episode) # go to the next state, explore children

            # RL learns from this transition and calculating the cost
            cost += RL.learn(str(observation), action, reward, str(observation_))

            #Keep track of transition: this state, next state and action that took us to next state
            if observation != None:
                current_state = observation.copy()
            else:
                current_state = None
            if observation_ not in ['goal','forest']:
                next_state = observation_.copy()
            else:
                next_state = observation_

            #store current and next state to be used for later
            if i == 0:
                transition_tracker[i] = [current_state,next_state,action]
            else:
                transition_tracker[i] = [transition_tracker[i-1][1],next_state,action] #current state is the previous next from last iteration

            # Swapping the observations - current and next
            observation = observation_

            # Calculating number of Steps in the current Episode
            i += 1

            # Break while loop when it is the end of current Episode
            # When agent reached the goal or obstacle
            if done:
                steps += [i]
                all_costs += [cost]
                RL.backpropagate_reward(transition_tracker,reward)
                break
        #print()

    # Showing the final route
    env.final()

    # Showing the Q-table with values for each action
    RL.print_q_table(qtable_path)

    # Plotting the results
    #print(steps,all_costs)
    #RL.plot_results(steps, all_costs)


# Commands to be implemented after running this file
if __name__ == "__main__":
    #parse Parameters
    params = argparser()

    protein = params['protein']
    no_of_chains = params['nofchains']
    chains = list(params['chains'])
    clash_thres = params['clash_threshold']
    episode_no = params['episodes']
    data_dir = params['path']
    decoy_size = params['pool_size']

    #eps_list = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    lr_list = [0.01,0.001,0.0001,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    

    gridstart_time =  time.time()
    ln = data_dir + '/data/' + protein + '/' + protein + '_gridsearch.out'
    gridlog = data_dir + '/data/' + protein + '/' + protein + '_gridresults.csv'
    logging.basicConfig(filename=ln,filemode='a',format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',datefmt='%H:%M:%S',level=logging.INFO)
    gridlogger = logging.getLogger("MCRLGridLogger")
    gridlogger.info("Starting Multiple Docking Gridsearch for " + protein + " chain_list : " + str(chains))
    gridlogger.info("lr_list : " + str(lr_list))
    gridlogger.info("eps_list : " + str(eps_list))

    lr_results = {}

    for lrate in lr_list:
        lr_results[lrate] = []
        for eps in eps_list:
            epsilon = eps
            lr = lrate
            
            gridlogger.info('Starting Docking run using learning_rate : ' + str(lr) + ' epsilon : ' + str(epsilon))
            
            #logname = data_dir + '/data/' + protein + '/' + protein + '_' + str(lr) + '_' + str(eps) + '_mcrl.out'
            qtable_path = data_dir + '/data/' + protein + '/gridout/' + protein + '_' + str(lr) + '_' + str(eps) + '_qtable.csv'

            gridlogger.info("Starting Multiple Docking for " + protein + " chain_list : " + str(chains)+  ' learning_rate : ' + str(lr) + ' epsilon : ' + str(eps))
            print("Starting Multiple Docking for " + protein + " chain_list : " + str(chains)+  ' learning_rate : ' + str(lr) + ' epsilon : ' + str(eps))
            start_time = time.time()
            gridlogger.info("Docking Parameters: Clash Threshold = " + str(clash_thres) + " Total Episodes : " + str(episode_no))
            mylogger = logging.getLogger("mcrllogger")


            env = Environment(chain_length=no_of_chains,chain_list=chains,protein_name=protein,thresh=clash_thres, epsln=epsilon, log=mylogger, learning_rate = lr, path=data_dir)
            RL = QLearningTable(actions=list(range(env.n_actions)),e_greedy=epsilon,log=mylogger,learning_rate = lr)
            update(episode_no,qtable_path)


            found_energies = env.protein.energy_tragetory
            # mylogger.info("Energy_tragetory for lr : " + str(lr) + " eps : "+ str(epsilon) +" : " + str(found_energies))
            gridlogger.info("Energy_tragetory for lr : " + str(lr) + " eps : "+ str(epsilon) +" : " + str(found_energies))
            
            lr_results[lr].append(min(found_energies)) # keep track of the best energy found

            total_time = time.time() - start_time

            gridlogger.info("Docking for  " + protein + " chain_list : " + str(chains) +  ' learning_rate : ' + str(lr) + ' epsilon : ' + str(eps) + " Completed, Total running time : " + str(total_time))
            print("Docking for  " + protein + " chain_list : " + str(chains) +  ' learning_rate : ' + str(lr) + ' epsilon : ' + str(eps) + " Completed, Total running time : " + str(total_time))


        gridlogger.info("Finished Gridsearch for learning_rate : " + str(lr))

    
    print()
    print(lr_results)
    result = pd.DataFrame(lr_results)
    print(result)
    row, col = result.stack().idxmin()
    gridlogger.info("Best parameter found using Gridsearch is : epsilon : " + str(eps_list[row]) + " learning_rate: " + str(col) + " which resulted in energy : " +str(result.loc[row,col]))

    result.to_csv(gridlog)

    gridtotal_time = time.time() - gridstart_time
    gridlogger.info("Gridsearch for all parameters Completed, Total running time : " + str(gridtotal_time))
