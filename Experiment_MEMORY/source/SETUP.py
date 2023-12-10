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

import itertools as it

#Enum Settings
#===================================================================================================
# vzd.Mode:
# -PLAYER
# -SPECTATOR
# -ASYNC_PLAYER
# -ASYNC_SPECTATOR

#vzd.ScreenFormat:
# CRCGCB - 3 channels of 8-bit values in RGB order
# RGB24 - channel of RGB values stored in 24 bits, where R value is stored in the oldest 8 bits
# RGBA32 - channel of RGBA values stored in 32 bits, where R value is stored in the oldest 8 bits
# ARGB32 - channel of ARGB values stored in 32 bits, where A value is stored in the oldest 8 bits
# CBCGCR - 3 channels of 8-bit values in BGR order
# BGR24 - channel of BGR values stored in 24 bits, where B value is stored in the oldest 8 bits
# BGRA32 - channel of BGRA values stored in 32 bits, where B value is stored in the oldest 8 bits
# ABGR32 - channel of ABGR values stored in 32 bits, where A value is stored in the oldest 8 bits
# GRAY8 - 8-bit gray channel
# DOOM_256_COLORS8 - 8-bit channel with Doom palette values

#vzd.ScreenResolution:
# RES_160X120   RES_400X300   RES_1024X768
# RES_200X125   RES_512X288   RES_1280X720
# RES_200X150   RES_512X320   RES_1280X800
# RES_256X144   RES_512X384   RES_1280X960
# RES_256X160   RES_640X360   RES_1280X1024
# RES_256X192   RES_640X400   RES_1400X787
# RES_320X180   RES_640X480   RES_1400X875
# RES_320X200   RES_800X450   RES_1400X1050
# RES_320X240   RES_800X500   RES_1600X900
# RES_320X256   RES_800X600   RES_1600X1000
# RES_400X225   RES_1024X576  RES_1600X1200
# RES_400X250   RES_1024X640  RES_1920X1080

#vzd.AutomapMode:
# Enum type that defines all automapBuffer modes.
# NORMAL - Only level architecture the player has seen is shown.
# WHOLE - All architecture is shown, regardless of whether or not the player has seen it.
# OBJECTS - In addition to the previous, shows all things in the map as arrows pointing in the direction they are facing.
# OBJECTS_WITH_SIZE - In addition to the previous, all things are wrapped in a box showing their size.
     
# vzd.GameVariable
# Enum type that defines all variables that can be obtained from the game.

# vzd.Button
# Enum type that defines all buttons that can be “pressed” by the agent.




#Creates a Game Given a Config File and Set Parameters
#===================================================================================================
def createGame(config_file_path):
    print(f"Setting Up Doom Game")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game = setGameParameters(game)
    game.init()
    print("==================================================================================================")
    return game



#Creates a Game Given a Config File and Set Parameters
#===================================================================================================
def setGameParameters(game):

    game.set_window_visible(False)
    #game.set_window_visible(True)

    game.set_depth_buffer_enabled(True)
    # game.set_automap_buffer_enabled(True)
    # game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    # game.set_automap_rotate(False)
    # game.set_automap_render_textures(False)

    game.set_mode(vzd.Mode.PLAYER)
    #game.set_mode(vzd.Mode.SPECTATOR)
    #game.set_mode(vzd.Mode.ASYNC_PLAYER)
    #game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
    
    game.set_screen_format(vzd.ScreenFormat.GRAY8)

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    game.set_render_decals(False)

    game.set_render_hud(True)

    game.set_render_corpses(False)

    # # Enables buffer with top down map of he current episode/level .
    # game.set_automap_buffer_enabled(True)
    # game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    # game.set_automap_rotate(False)
    # game.set_automap_render_textures(False)

    return game



#Generate Action Space
#===================================================================================================
def getActionSpace(game):
    n = game.get_available_buttons_size()
    print(f"Number of Buttons: {n}")

    #All Possible Button Combinations
    #If there are 3 buttons there are 2*3 = 8 combinations of actions
    # [0, 0, 0]
    # [0, 0, 1]
    # [0, 1, 0]
    # [0, 1, 1]
    # [1, 0, 0]
    # [1, 0, 1]
    # [1, 1, 0]
    # [1, 1, 1]
    #actions = [list(a) for a in it.product([0, 1], repeat=n)]

    #All Possible Button Combinations (Omit No Action)
    # [0, 0, 1]
    # [0, 1, 0]
    # [0, 1, 1]
    # [1, 0, 0]
    # [1, 0, 1]
    # [1, 1, 0]
    # [1, 1, 1]
    # actions = [list(a) for a in it.product([0, 1], repeat=n)][1:]


    # All Possible Button Combinations (Omit No Action and All Actions)
    # [0, 0, 1]
    # [0, 1, 0]
    # [0, 1, 1]
    # [1, 0, 0]
    # [1, 0, 1]
    # [1, 1, 0]
    # [1, 1, 1]
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    actions = actions[1:len(actions)-1]


    # # Limit to Pressing 1 Button at A Time (Omits No Action)
    # # actions = [[1,0,0],[0,1,0],[0,0,1]]
    # actions = []
    # for button in range(n):
    #     combo = [0]*n
    #     combo[button] = 1
    #     actions.append(combo)

    return actions
