"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
#import imp
#test
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method
from LLMQueryEnv import LLMQueryEnv
from mcts import execute_episode,test_episode,initialize_MCTS_tree
import argparse,os,re
import os.path as osp
import torch,shutil
import statistics,pickle
from mcts import MCTS
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MCTS+LLM')
    parser.add_argument('--init_prompt', type=str, required=False,
                        help='Initial prompt')
    parser.add_argument('--dumpdir', type=str, required=True,default="",
                        help='DUMP directory for storing result')
    parser.add_argument('--runID', type=int, required=True, default=0,
                        help='Run ID')
    parser.add_argument('--sims', type=int, required=False, default=1000,
                        help='Simulations per MCTS episode')
    parser.add_argument('--ep', type=int, required=False, default=50,
                        help='#MCTS episode')
    args = parser.parse_args()
    
    origPrompt = args.init_prompt
    rootDumpDir = args.dumpdir
    runID = args.runID
    simulation_per_episode = args.sims
    num_episodes = args.ep
    prompt_str = ""
    module_name = ""
    prob_files_path = ""
    set_start_method("spawn")

    runFolder = osp.join(rootDumpDir,'run'+str(runID))
    csvResultFile = osp.join(rootDumpDir,'data_run'+str(runID)+".csv") # Data to store area obtained in each iteration and best so far
    logFilePath = osp.join(runFolder,'log_run'+str(runID)+".log")   # Log data to dump area and read
    if not osp.exists(rootDumpDir):
        os.mkdir(rootDumpDir)
        
    if not osp.exists(runFolder):    
        os.mkdir(runFolder)

    csvFileHandler = open(csvResultFile,'w')
    
    if not origPrompt:
        #-----------Alter the two lines below for your prompt files.---------------#
        #TODO
        #------For now, file_name (without the .v) should be the same as module name in prompt file for code to run.
        file_dir = "VGen-main/prompts-and-testbenches/intermediate2"
        file_name = "prompt3_counter.v"

        #Specify your prompt file to use as input.
        promptfile_path = file_dir + "/" + file_name
        #Reading input file data and finding the module name.
        #Parsing the file name for the problem name (counter, lfsr, etc).
        file_name_parts = file_name.split("_", 1)
        if len(file_name_parts) > 1:
            problem_name = "_".join(file_name_parts[1:])
            print("Joined problem name: ", problem_name)
            if problem_name.endswith(".v"):
                    problem_name = problem_name.replace(".v","")
        else:
            print("Error parsing file name - check formatting (ex: prompt1_counter.v)")
            exit(1)

        try:
            with open(promptfile_path, "r") as prompt_file:
                prompt_str = prompt_file.read()
        except:
            print("Error reading prompt file.")
            exit(1)

    idx_ep = 0
    mctsTree = initialize_MCTS_tree(LLMQueryEnv(orig_prompt=prompt_str, orig_module=problem_name, file_path=file_dir))
    #print("Episode not stated yet!")
    #while idx_ep<num_episodes:
    #    with ProcessPoolExecutor(max_workers=simulation_per_episode) as executor:
    #        futures = [executor.submit(execute_episode, mctsTree, 1) for _ in range(simulation_per_episode)]
    #        #results = [future.result() for future in futures]
    #    idx_ep+=1  
    #print(mctsTree.num_simulations)
    
    
    
    while idx_ep<num_episodes:
        print("******** EPISODE-{}************".format(idx_ep+1))
        mctsTree = execute_episode(mctsTree,simulation_per_episode)
        evalMctsMaxValue,evalMctsRobustValue = test_episode(mctsTree)
        csvFileHandler.write("{},{}\n".format(evalMctsMaxValue,evalMctsRobustValue))
        idx_ep+=1
    
