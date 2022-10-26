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

```$ python3 main.py```

if you want to train the saved model, choose the number of games(**n_games**) you want to train, and number of model(**md_num**)

(for examplethe number for the model "model_until_200epoch.model" is 200)

and run main.py as 

```$ python3 main.py -c -n n_games -md md_num```

then it will train continuosly.

## model

Used simple DQN(Deep Q Network) with 4 FCs(Fully Connected Networks)

E-greedy method with epsilon_max = 0.9 & epsilon_min = 0.01 and epsilon_dec = 0.996
(next_eps = eps / eps_dec)

replay memories are stored in Agent as Agent.state_memory, Agent.new_state_memory ... 

## Visualization / Images of score

![Screenshot from 2019-12-01 20-47-52](https://user-images.githubusercontent.com/47442084/69914075-49d21f00-1483-11ea-9b98-4ddd91bc059b.png)
![Screenshot from 2019-12-01 20-47-28](https://user-images.githubusercontent.com/47442084/69914074-49398880-1483-11ea-8e85-deb42a748439.png)
![first 200 epoch scores](https://user-images.githubusercontent.com/47442084/69914077-4ccd0f80-1483-11ea-8ebc-4bc7144cbd28.jpg)
![scores200~250](https://user-images.githubusercontent.com/47442084/69914078-4e96d300-1483-11ea-8a6c-9beaa52aa4ec.jpg)
![scores250~300](https://user-images.githubusercontent.com/47442084/69914079-4e96d300-1483-11ea-8bb9-9c9cc961c501.jpg)
![rewards](https://user-images.githubusercontent.com/47442084/69914081-4fc80000-1483-11ea-8861-5db473f236e3.jpg)
![steps](https://user-images.githubusercontent.com/47442084/69914083-5191c380-1483-11ea-8186-aae593fdec8a.jpg)

