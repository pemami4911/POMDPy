__author__ = 'patrickemami'

import os

# important directories
CFG_DIR = 'config'
ROCK_CFG_FILE = 'rockproblem-config.json'
SYS_CFG_FILE = 'system-config.json'
LOG_DIR = 'log'
LOG_FILE = 'POMDPy.log'

dir = os.path.dirname(__file__)
rock_cfg = os.path.join(dir, '..', CFG_DIR, ROCK_CFG_FILE)
sys_cfg = os.path.join(dir, '..', CFG_DIR, SYS_CFG_FILE)
log_path = os.path.join(dir, '..', LOG_DIR, LOG_FILE)

def parse_map(map):
    map_text = []

    with open(os.path.join(dir, '..', CFG_DIR, map), "r") as f:
        dimensions = f.readline().strip().split()
        for line in f:
            map_text.append(line.strip())
    return tuple([map_text, dimensions])
