import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd

import sys
sys.path.insert(1, f'{sys.path[0]}/source')

from AGENT import DQNAgent
from RUN_AND_TEST import run
from SETUP import createGame, getActionSpace
from WATCH_AND_RECORD import watch, record, load_record
from CONFIG import *

########################################################################################################################################################################################################


if __name__ == "__main__":

    memories_to_load = ["data/basic_two_cacodemon_MEMORY.data", "data/basic_one_wolfensteinSS_MEMORY.data"]

    #Parameters Loaded from Config File 

    #Loop Through Every Task
    #=======================================================================================================================
    for task_index in range(len(scenarios)):

        # learning_steps_per_epoch = starting_learning_steps_per_epoch*(2**task_index)

        # if task_index != 0:
        #     load_model = True
    
        #Create a new game 
        cfg_file = scenarios[task_index]
        model_savefile = scenario_save_file
        game = createGame(cfg_file)
        result_id = result_ids[task_index]

        print()
        print(f"Task: {cfg_file}")

        #get action space 
        actions = getActionSpace(game)

        # Initialize our agent with the set parameters
        agent = agent_model(
            action_size = len(actions),
            memory_size = replay_memory_size,
            batch_size = batch_size,
            discount_factor=discount_factor,
            lr=learning_rate,
            load_model=load_model,
            model_savefile=model_savefile,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            MEMORY_FILE=MEMORY_FILE
        )

        for mem_file in memories_to_load:
            agent.load_memory(mem_file)

        # Run the training for the set number of epochs
        if not skip_learning:
            agent, game = run(
                result_id=result_id,
                game=game, 
                agent=agent, 
                actions=actions, 
                num_epochs=train_epochs, 
                frame_repeat=frame_repeat, 
                resolution=resolution,
                save_model=save_model, 
                model_savefile=model_savefile, 
                test_episodes_per_epoch=test_episodes_per_epoch, 
                steps_per_epoch=learning_steps_per_epoch
                )

        # Reinitialize the game with window visible
        game.close()
        
        # watch(game,agent,actions,frame_repeat,resolution,episodes_to_watch)
        record(game,agent,actions,frame_repeat,resolution,episodes_to_watch,task_index)
        # load_record(game,"recordings/Episode_0-6.lump")
        



