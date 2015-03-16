__author__ = 'patrickemami'

# Parse the configuration parameters from config/config.json

# needed for creating the path to access the config file
import os

# file directory and name
CFG_DIR = "config"
CFG_FILE = "config.json"

dir = os.path.dirname(__file__)
cfg_file = os.path.join(dir, '..', CFG_DIR, CFG_FILE)

def parse_map(map):
    map_text = []

    with open(os.path.join(dir, '..', CFG_DIR, map), "r") as f:
        dimensions = f.readline().strip().split()
        for line in f:
            map_text.append(line.strip())
    return tuple([map_text, dimensions])
