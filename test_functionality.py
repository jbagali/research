import numpy as np
import gym # open ai gym
import os,re,shutil,argparse
import time
import torch
import random
import subprocess
from static_env import StaticEnv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
testbench_path = "test_tb.v"
module_path = "test_module.v"

compile_output = subprocess.check_output(['iverilog', '-o', str(os.getpid()) + '_simulation', testbench_path, module_path], stderr=subprocess.STDOUT)
try:
    sim_path =str(os.getpid()) + '_simulation'
    simulation_output = subprocess.check_output(['vvp', sim_path], stderr=subprocess.STDOUT)
    simulation_exit_code = 0
    print(simulation_output)
except subprocess.CalledProcessError as e:
    simulation_output = e.output
    simulation_exit_code = e.returncode
    print(simulation_output)