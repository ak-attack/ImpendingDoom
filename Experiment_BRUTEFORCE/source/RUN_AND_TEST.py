import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import skimage.transform

import os
import sys

import itertools as it
import random
from collections import deque
from time import sleep, time
from tqdm import trange

import vizdoom as vzd


from PROCESS import preprocess


#Set Device To Use (CPU or GPU)
#===================================================================================================
def setDevice():
    # Uses GPU if available
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print(f"DEVICE: {DEVICE}")
    return DEVICE


#Test Agent
#===================================================================================================
def test(game, agent, actions, frame_repeat, resolution, test_episodes_per_epoch):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print()
    print("Testing")
    print("==================================================================================================")

    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():

            state = preprocess(game.get_state().screen_buffer,resolution)
            best_action_index = agent.get_action(state)
            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(f"""
        Mean Reward:            {test_scores.mean()}
        Standard Deviation:  +/-{test_scores.std()}
        Minimum Reward:         {test_scores.min()}
        Maximum Reward:         {test_scores.max()}
        """)



#Run Number of Training Epochs
#===================================================================================================
def run(result_id, game, agent, actions, num_epochs, frame_repeat, resolution, save_model, model_savefile, test_episodes_per_epoch, steps_per_epoch):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """
    print()
    print("Running")
    print("==================================================================================================")

    mean_reward = []
    standard_deviation = []
    min_reward = []
    max_reward = []

    start_time = time()
    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print(f"\nEpoch #{epoch + 1}")

        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess(game.get_state().screen_buffer,resolution)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            reward = reward - (100-game.get_game_variable(vzd.HEALTH))
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer,resolution)
            else:
                next_state = np.zeros((1, resolution[0], resolution[1])).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()
            #agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)

        mean_reward.append(train_scores.mean())
        standard_deviation.append(train_scores.std())
        min_reward.append(train_scores.min())
        max_reward.append(train_scores.max())

        print(f"""
            Mean Reward:            {train_scores.mean()}
            Standard Deviation:  +/-{train_scores.std()}
            Minimum Reward:         {train_scores.min()}
            Maximum Reward:         {train_scores.max()}
            """)

        test(game, agent, actions, frame_repeat, resolution, test_episodes_per_epoch)

        if save_model:
            print(f"Saving Network Weights to {model_savefile}")
            torch.save(agent.q_net, model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    filename = f"results/{result_id}_Rewards"
    result_list = np.array([mean_reward,standard_deviation,min_reward,max_reward])

    np.save(filename, result_list)

    game.close()
    return agent, game







