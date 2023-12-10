import pickle

import os
from argparse import ArgumentParser
from random import choice

import cv2
import numpy as np
import skimage.transform

import vizdoom as vzd
from collections import deque

import itertools as it

import sys
sys.path.insert(1, f'{sys.path[0]}/source')

from PROCESS import preprocess
from SETUP import setGameParameters, getActionSpace
from CONFIG import resolution, scenarios, MEMORY_FILE


CONFIG = scenarios[0]
memory = deque(maxlen=10000)


def write_memory():
    with open(MEMORY_FILE, "wb") as fp:  
        pickle.dump(memory, fp)


def read_memory():
    loaded_mem = []
    with open(MEMORY_FILE, "rb") as fp:
        loaded_mem = pickle.load(fp)
    return loaded_mem



if __name__ == "__main__":

    #Create a new game (ensure that all parameters match the experiment)
    print(f"Setting Up Doom Game")
    game = vzd.DoomGame()
    game.load_config(CONFIG)
    game = setGameParameters(game)

    # # OpenCV uses a BGR colorspace by default.
    # game.set_screen_format(vzd.ScreenFormat.BGR24)
    # game.set_screen_format(vzd.ScreenFormat.GRAY8)

    # # Sets resolution for all buffers.
    # game.set_screen_resolution(vzd.ScreenResolution.RES_1280X1024)

    # # Enables depth buffer.
    # game.set_depth_buffer_enabled(True)

    # # Enables labeling of in game objects labeling.
    # game.set_labels_buffer_enabled(True)

    # # Enables buffer with top down map of he current episode/level .
    # game.set_automap_buffer_enabled(True)
    # game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    # game.set_automap_rotate(False)
    # game.set_automap_render_textures(False)

    # game.set_render_hud(True)
    # game.set_render_minimal_hud(False)

    game.set_window_visible(True)

    game.set_mode(vzd.Mode.SPECTATOR)
    game.init()


    #get action space
    actions = getActionSpace(game)

    #Note: data collected from first episode will be omitted 
    #in order to give you time to prepare/adjust to playing.
    episodes = 101
    sleep_time = 0.028

    for i in range(episodes):
        total_reward = 0
        print(f"###########################################################################################")

        print(f"Episode #{i + 1} : START")

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():
            #gets the entire state object
            state = game.get_state()

            # Display all the buffers here!

            # Just screen buffer, given in selected format. This buffer is always available.
            screen = state.screen_buffer
            cv2.imshow("ViZDoom Screen Buffer", screen)

            # # Depth buffer, always in 8-bit gray channel format.
            # # This is most fun. It looks best if you inverse colors.
            # depth = state.depth_buffer
            # if depth is not None:
            #     cv2.imshow("ViZDoom Depth Buffer", depth)

            # # Labels buffer, always in 8-bit gray channel format.
            # # Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
            # # Labels data are available in state.labels, also see labels.py example.
            # labels = state.labels_buffer
            # if labels is not None:
            #     cv2.imshow("ViZDoom Labels Buffer", labels)

            # # Map buffer, in the same format as screen buffer.
            # # Shows top down map of the current episode/level.
            # automap = state.automap_buffer
            # if automap is not None:
            #     cv2.imshow("ViZDoom Map Buffer", automap)

            #Memory Data Format : (state, action, reward, next_state, done)
            state = preprocess(state.screen_buffer,resolution)
            game.advance_action()
            action = game.get_last_action()
            reward = game.get_last_reward()
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer,resolution)
            else:
                next_state = np.zeros((1, resolution[0], resolution[1])).astype(np.float32)

            reward = reward - (100-game.get_game_variable(vzd.HEALTH))
            total_reward += reward

            print(f"Episode {i+1} : {game.get_episode_time()}/{game.get_episode_timeout()}")
            print(f"Memory : {len(memory)}")
            print(f"Total Reward : {total_reward}")

            if len(memory) >= 10000-1:
                break

            if i != 0 and action in actions:
                print("appending action")
                action_index = actions.index(action)
                memory.append( (state, action_index, reward, next_state, done) )
        
            cv2.waitKey(int(sleep_time * 1000))

        print(f"Episode #{i + 1} : END")

    #write to memory file
    write_memory()
    loaded_mem = read_memory()

    cv2.destroyAllWindows()

    print("************************************************************************************************")
    print(f"Total Number of Data points: {len(loaded_mem)}")

