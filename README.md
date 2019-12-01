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

you can adjust the amount or conditions oh these rewards in the game_environment and it will affect the training efficiency a lot.

## How to run

first, run main.py with no argumets. The model will be trained until 200 epochs and be saved as "model_until_200epoch.model"

$ python3 main.py

if you want to train the saved model, choose the number of games(n_games) you want to train, and number of model(md_num)

(for examplethe number for the model "model_until_200epoch.model" is 200)

and run main.py as 

$ python3 main.py -c -n n_games -md md_num

then it will train continuosly.

## model

Used simple DQN(Deep Q Network) with 4 FCs(Fully Connected Networks)

E-greedy method with epsilon_max = 0.9 & epsilon_min = 0.01 and epsilon_dec = 0.996
(next_eps = eps / eps_dec)

replay memories are stored in Agent as Agent.state_memory, Agent.new_state_memory ... 

## Visualization / Images of score

![Screenshot from 2019-12-01 19-42-01](https://user-images.githubusercontent.com/47442084/69913188-dd9dee00-1477-11ea-95c2-9313734d5825.png)
![Screenshot from 2019-12-01 19-42-11](https://user-images.githubusercontent.com/47442084/69913189-dd9dee00-1477-11ea-88f9-99f6baa44ade.png)
![scores](https://user-images.githubusercontent.com/47442084/69913191-e262a200-1477-11ea-8734-dd18a943de44.jpg)
![scores200~250](https://user-images.githubusercontent.com/47442084/69913192-e2fb3880-1477-11ea-9012-0550be24e0ba.jpg)
![scores250~300](https://user-images.githubusercontent.com/47442084/69913193-e2fb3880-1477-11ea-9c82-ba30bb322f46.jpg)
