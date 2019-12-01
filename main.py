from model.model import *
import argparse
import numpy as np
import torch
import sys
import time
from util.utils import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
#from gym import wrappers

if __name__ == '__main__':
    lr = 0.001
    brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                input_dims=[4], lr=lr)
    n_games = 200
    start_epoch = 0
    md_num = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--continue_training", action="store_true")
    parser.add_argument("-n","--n_games", type=int)
    parser.add_argument("-md","--md_number", type=int)
    args = parser.parse_args()
    memories = None
    if args.continue_training :
        n_games = args.n_games
        md_num = args.md_number
        start_epoch = md_num
        brain = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=4,
                    input_dims=[4], lr=lr)
        mdname = "model_until_%depoch.model" % md_num
        brain.Q_eval, brain.Q_eval.optimizer, memories = load_model(mdname, 4, brain.Q_eval.device, 0.001)
        brain.state_memory = memories[0]
        brain.new_state_memory = memories[1]
        brain.action_memory = memories[2]
        brain.reward_memory = memories[3]
        brain.terminal_memory = memories[4]
        #exit()
    eps_history = []
    scores = []
    rewards = []
    score = 0
    #scheduler = MultiStepLR(brain.Q_eval.optimizer, milestones=[60,80], gamma=0.5)
    steps = []
    for i in range(start_epoch, start_epoch + n_games):
        env = game_env(i)
        score = 0
        eps_history.append(brain.epsilon)

        observation = env.reset(env.state)
        done = False
        step = 0
        r = 0
        while not done:
            env.visualize()
            t = time.time()
            action = brain.choose_action(observation)
            observation_, reward, done = env.step(action, step)
            #score += reward

            print("score:",score, "reward", reward)

            brain.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            print("state", env.state)
            brain.learn()
            step += 1
            r += reward
            if reward == 10:
                observation = env.reset(env.state)
                steps.append(step)
                step = 0
                score += 1
        if reward != 10:
            r += reward
            steps.append(step)
            #print("time", time.time() - t)
            #print(observation_)
        scores.append(score)
        rewards.append(r)
        #scheduler.step()
    md_num += n_games
    mdname = "model_until_%depoch.model" % md_num
    save_model(mdname, brain, brain.Q_eval, brain.Q_eval.optimizer)

    print("average score:", np.average(scores))
    print("max score:", max(scores))
    if args.continue_training :
        plt.plot(np.arange(start_epoch, start_epoch + n_games),scores)
        plt.title("scores{}~{}.jpg".format(start_epoch, start_epoch + n_games))
        plt.savefig("scores{}~{}.jpg".format(start_epoch, start_epoch + n_games))
        plt.show()
    else:
        plt.plot(scores)
        plt.title("first %d epoch scores.jpg"%n_games)
        plt.savefig("first %d epoch scores.jpg"%n_games)
        plt.show()
        plt.plot(steps)
        plt.title("steps.jpg")
        plt.savefig("steps.jpg")
        plt.show()
        plt.plot(rewards)
        plt.title("rewards.jpg")
        plt.savefig("rewards.jpg")
        plt.show()
