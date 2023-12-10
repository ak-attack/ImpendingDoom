# ImpendingDoom
Final project for CS175!

### Originally Run On:
* Python Version: 3.10.12
* Operating System: Pop!_OS 22.04 LTS x86_64
* Hardware: CPU

### Dependancies:
- VizDoom (used version 1.2.2):       ```pip install vizdoom```
- Numpy (used version 1.23.5):        ```pip install numpy```
- Matplotlib (used version 3.5.1):    ```pip install matplotlib```
- PyTorch (used version 2.1.1):       ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```
- Opencv (used version 4.8.1.78):     ```pip install opencv-python```
- scikit-image (used version 0.22.0): ```pip install scikit-image```

## Code Organization
Experiments were divided into separate folders. Each experiment is matched to its corresponding folder below:
- Brute Force Experiment:               ```Experiment_BRUTEFORCE```
- Iterative Learning Experiment:        ```Experiment_ITERATIVE```
- Imitative Initialization Experiment : ```Experiment_IMITATIVE```
- Memory Expansion Experiment:          ```Experiment_MEMORY```
- Ensemble Experiment:                  ```Experiment_ALL```

Depending on what was tested in the experiment, the following folders and files were included: 
- Folders:
  - Data - any imitation data or preserved memory used
  - Recordings - any recorded episodes
  - Scenarios - .cfg and associated .wad files
  - Results - Numpy data needed for plotting results
  - Source
    - AGENT - the model used (Duel DQN)
    - NETWORK - the neural network used  (Dueling NN)
    - CNN_CALCUATIONS - convolutional neural network calculations
    - PROCESS - game screen state processing functions
    - RUN_AND_TEST - functions for running training/testing on the agent
    - SETUP - functions for setting up the VizDoom environment
    - WATCH_AND_RECORD - functions for watching episodes, recording episodes, and loading recorded episodes
- Files:
  - EXPERIMENT_DESCRIPTION.txt - detailed description of the experiment contained in the given folder
  - Experiment.py - Python script to run
  - gatherData.py - Python script used to collect human player data
  - CONFIG.py - contains experiment parameters
  - .pth Files - save PyTorch model state dictionary
  - .out Files - redirected experiment output

You can find all graphs plotted in PlotData.ipynb.

## Run a Pretrained Model:
- Brute Force Experiment:               ```python3 Experiment_BRUTEFORCE/Experiment.py > Experiment.out```
- Iterative Learning Experiment:        ```python3 Experiment_ITERATIVE/Experiment.py > Experiment.out```
- Imitative Initialization Experiment : ```python3 Experiment_IMITATIVE/Experiment.py > Experiment.out```
- Memory Expansion Experiment:          ```python3 Experiment_MEMORY/Experiment.py > Experiment.out```
- Ensemble Experiment:                  ```python3 Experiment_ALL/Experiment.py > Experiment.out```

## Train a Model From Scratch:
In order to train a model from scratch you will have to alter the CONFIG.py file of the experiment you want to run.

Each CONFIG.py file stores these variables:
```
save_model = False
load_model = True
skip_learning = True
```
They must be must be swapped to these values in order for the experiment to run from scratch:
```
save_model = True
load_model = False
skip_learning = False
```
Afterwards you can run the experiment like so:

- Brute Force Experiment:               ```python3 Experiment_BRUTEFORCE/Experiment.py > Experiment.out```
- Iterative Learning Experiment:        ```python3 Experiment_ITERATIVE/Experiment.py > Experiment.out```
- Imitative Initialization Experiment : ```python3 Experiment_IMITATIVE/Experiment.py > Experiment.out```
- Memory Expansion Experiment:          ```python3 Experiment_MEMORY/Experiment.py > Experiment.out```
- Ensemble Experiment:                  ```python3 Experiment_ALL/Experiment.py > Experiment.out```

