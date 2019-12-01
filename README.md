# DQN_with_grid_world_bait_eating_simulator

## Environment

Ubuntu 18.04
pytorch 1.2.0

## simulation of grid world for reinforcement learning

Environment class for grid world is in util/utils.py as a class of game_env()

the grid world is formed with 4 x 6 grids and init_state, goal_state, current_state.

In this game, the Agent will choose wether to go up, down, right, left to go to goal

if the agent reaches goal, it will receive + rewards and the next game will begin and the initial state will be the last goal state

if Agent gets out of the grid world, it becomes game over and Agent will recieve - rewards.

## How to train

first, run main.py with no argumets. The model will be trained until 200 epochs and be saved as "model_until_200epoch.model"

$ python3 main.py

if you want to train the saved model, choose the number of games(n_games) you want to train, and number of model(md_num)

(for examplethe number for the model "model_until_200epoch.model" is 200)

and run main.py as 

$ python3 main.py -c -n n_games -md md_num

then it will train continuosly.

##Visualization / Images of score
