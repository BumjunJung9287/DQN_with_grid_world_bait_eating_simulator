import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.model import DeepQNetwork

class game_env():
    def __init__(self, i):
        self.episod = i
        self.obstical = []
        self.init_state = random_position_except(self.obstical)
        self.obstical.append(self.init_state)
        self.goal = random_position_except(self.obstical)
        self.obstical.append(self.goal)
        self.state = self.init_state[:]
        self.action = None
        self.start = True


    def reset(self, init_state):
        self.obstical = []
        self.init_state = init_state[:]
        self.obstical.append(self.init_state)
        self.goal = random_position_except(self.obstical)
        self.obstical.append(self.goal)
        self.state = self.init_state[:]
        self.action = None
        self.start = True

        self.observation = []
        observation = [self.state, self.goal]
        for slist in observation:
            for s in slist:
                self.observation.append(s) # currenct state + goal_state
        return self.observation

    def step(self, action, step):
        self.action = action
        self.last_state = self.state[:]
        if action == 0:
            self.state[1] += 1
        elif action == 1:
            self.state[1] -= 1
        elif action == 2:
            self.state[0] -= 1
        elif action == 3:
            self.state[0] += 1

        self.observation = []
        observation = [self.state, self.goal]
        for slist in observation:
            for s in slist:
                self.observation.append(s) # currenct state + goal_state

        reward = 0
        last_distance = np.linalg.norm(np.array(self.last_state) - np.array(self.goal))
        current_distance = np.linalg.norm(np.array(self.state) - np.array(self.goal))
        reward = (last_distance - current_distance)
        print("distances",last_distance, current_distance)
        print("distance_reward", reward)
        reward -= step
        print("step:",step)
        if self.state == self.goal:
            self.reward = 10
        else:
            self.reward = reward
        if check_end(self.state):
            self.done = True
            self.reward = -10
        else:
            self.done = 0
        if step > 10:
            self.reward = -3
            self.done = True
            print("too late")
        self.start = False
        return self.observation, self.reward, self.done

    def visualize(self):
        visualize(self.init_state, self.goal, self.state, self.action, self.episod, start=self.start)


def random_position_except(obstical_list):
    while True:
        x = np.random.choice(np.arange(0,4))
        y = np.random.choice(np.arange(0,6))
        xy = list([x,y])
        if xy not in (obstical_list):
            return xy

def check_end(state):
    x = state[0]
    y = state[1]
    if x<0 or x>3 or y<0 or y>5:
        return True
    else:
        return False

def save_model(name, brain, dqn_net, optimizer):
    print("Saving model... wait...")
    memories = []
    memories.append(brain.state_memory)
    memories.append(brain.new_state_memory)
    memories.append(brain.action_memory)
    memories.append(brain.reward_memory)
    memories.append(brain.terminal_memory)

    torch.save(
        {
            "dqn": dqn_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "memories": memories
        },
        name,
    )
    print("Model saved!")

def load_model(md_name, n_actions, device, lr):
    dqn_net = DeepQNetwork(lr, [4], 256, 256, 100, 4).to(device)
    optimizer = optim.Adam(dqn_net.parameters(), lr=lr)
    memories = None
    try:
        checkpoint = torch.load(md_name)
        dqn_net.load_state_dict(checkpoint["dqn"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        memories = checkpoint["memories"]

        print("Models loaded!")
    except Exception as e:
        print(f"Couldn't load Models! => {e}")
    return dqn_net, optimizer, memories


def visualize(init_state, goal, state, last_action, episod, start=False):
    print("episod %d",episod)
    arrows = {0:"►", 1:"◄", 2:"▲", 3:"▼"}
    map = [["#" for i in range(6)] for j in range(4)]
    map[goal[0]][goal[1]] = "G"
    map[init_state[0]][init_state[1]] = "S"
    print("init_state:", init_state)
    if not start:
        map[state[0]][state[1]] = arrows[last_action]
    else:
        print("start game!")
    for i in range(4):
        string = ""
        for j in range(6):
            string += map[i][j]
            string += " "
        print(string[:-1])
        print()
    print("############")
    if goal == state:
        print("Goal Scored!! genterating next goal!")
    return


if __name__=="__main__":
    visualize((0,0),(3,5),(0,0),None,0,start=True)
    visualize((0,0),(3,5),(0,1),0,0)
    visualize((0,0),(3,5),(0,2),0,0)
    visualize((0,0),(3,5),(0,3),0,0)
    visualize((0,0),(3,5),(0,4),0,0)
    visualize((0,0),(3,5),(0,5),0,0)

    visualize((0,0),(3,5),(1,5),3,0)
    visualize((0,0),(3,5),(2,5),3,0)
    visualize((0,0),(3,5),(3,5),3,0)
