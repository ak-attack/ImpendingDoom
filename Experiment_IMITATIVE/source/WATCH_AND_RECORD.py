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


#Watch Expisodes
#===================================================================================================
def watch(game, agent, actions, frame_repeat, resolution, episodes_to_watch):
    print()
    print("Watching")
    print("==================================================================================================")

    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    total_score = 0
    for i in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()

        total_score += score
        print("Total score: ", score)
    
    game.close()

    print()
    print(f"Average Testing Reward : {total_score/episodes_to_watch}")


#Record Expisodes
#===================================================================================================
def record(game, agent, actions, frame_repeat, resolution, episodes_to_watch, file_id):
    print()
    print("Recording")
    print("==================================================================================================")

    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    total_score = 0
    for i in range(episodes_to_watch):
        episode_file = f"recordings/Episode_{file_id}-{i+1}.lump"
        print(episode_file)
        game.new_episode(episode_file)
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()

        total_score += score
        print("Total score: ", score)

    game.close()

    print()
    print(f"Average Testing Reward : {total_score/episodes_to_watch}")


#Load Recorded Expisode 
#===================================================================================================
def load_record(game, episode_file):
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    game.replay_episode(episode_file)
    while not game.is_episode_finished():
        s = game.get_state()
        # Use advance_action instead of make_action.
        game.advance_action()
        r = game.get_last_reward()

    game.close()
    
