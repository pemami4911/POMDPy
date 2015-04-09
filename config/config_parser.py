__author__ = 'patrickemami'

# Parse the configuration parameters from config/rockproblem-config.json

# needed for creating the path to access the config file
import os

# file directory and name
CFG_DIR = "config"
ROCK_CFG_FILE = "rockproblem-config.json"
SYS_CFG_FILE = "system-config.json"

dir = os.path.dirname(__file__)
rock_cfg = os.path.join(dir, ROCK_CFG_FILE)
sys_cfg = os.path.join(dir, SYS_CFG_FILE)

def parse_map(map):
    map_text = []

    with open(os.path.join(dir, map), "r") as f:
        dimensions = f.readline().strip().split()
        for line in f:
            map_text.append(line.strip())
    return tuple([map_text, dimensions])
