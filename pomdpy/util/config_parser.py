import os

# important directories
CFG_DIR = 'config'
ROCK_CFG_FILE = 'rock_sample_config.json'
LOG_DIR = 'log'
LOG_FILE = 'POMDPy.log'

my_dir = os.path.dirname(__file__)
rock_cfg = os.path.join(my_dir, '..', CFG_DIR, ROCK_CFG_FILE)
log_path = os.path.join(my_dir, '..', LOG_DIR, LOG_FILE)


def parse_map(m):
    map_text = []

    with open(os.path.join(my_dir, '..', CFG_DIR, m), "r") as f:
        dimensions = f.readline().strip().split()
        for line in f:
            map_text.append(line.strip())
    return tuple([map_text, dimensions])
