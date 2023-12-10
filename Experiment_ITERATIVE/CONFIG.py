from AGENT import DQNAgent
import vizdoom as vzd

agent_model = DQNAgent
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 7
learning_steps_per_epoch = 2000
replay_memory_size = 10000 + 2000 + 2000
batch_size = 64
test_episodes_per_epoch = 100
frame_repeat = 12
resolution = (50, 50)
episodes_to_watch = 15
save_model = False
load_model = True
skip_learning = True
epsilon = 1
epsilon_decay = 0.9996
epsilon_min = 0.1
scenarios = ["scenarios/basic_two_cacodemon.cfg","scenarios/basic_upgraded.cfg"]
scenario_save_file = "./Experiment.pth"
LOAD_DATA = False
MEMORY_FILE = "data/MEMORY_basic.data"
DEVICE = "cpu"
ScreenFormat = vzd.ScreenFormat.GRAY8
ScreenResolution = vzd.ScreenResolution.RES_640X480
result_ids = ["basic_two_cacodemon","basic_upgraded"]