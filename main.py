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

from LLMQueryEnv import LLMQueryEnv
from mcts import execute_episode,test_episode,initialize_MCTS_tree
import argparse,os,re
import os.path as osp
import torch,shutil
import statistics,pickle
from mcts import MCTS

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
    prompt = ""
    module_name = ""
    prob_files_path = ""
    
    runFolder = osp.join(rootDumpDir,'run'+str(runID))
    csvResultFile = osp.join(rootDumpDir,'data_run'+str(runID)+".csv") # Data to store area obtained in each iteration and best so far
    logFilePath = osp.join(runFolder,'log_run'+str(runID)+".log")   # Log data to dump area and read
    if not osp.exists(rootDumpDir):
        os.mkdir(rootDumpDir)
        
    if not osp.exists(runFolder):    
        os.mkdir(runFolder)

    csvFileHandler = open(csvResultFile,'w')
    
    if not origPrompt:
        prob_files_path = "VGen-main/prompts-and-testbenches/intermediate2"
        #Specify your prompt file to use as input.
        promptfile_path = prob_files_path + "/prompt1_counter.v"
        #Reading input file data and finding the module name.
        prompt = ""
        try:
            with open(promptfile_path, "r") as prompt_file:
                prompt = prompt_file.read()
                match = re.search(r'module\s+(\w+)\s*\(', prompt)
                if match:
                    module_name = match.group(1)
                    print("Module name: ", module_name)
                else:
                    print("Error finding module name in prompt file.")
                    exit(1)
        except:
            print("Error reading prompt file.")
            exit(1)

        #Test prompt that works (intermediate2 - counter) if file read is not working.
        # Note-- prompts seem to work best with 1 line of comments at the top, 
        #   and with the module definition lines 1 tab to the right below the comment.

                #prompt_test = "// This is a counter that counts from 1 to 12\n\
                #module counter(\n\
                #input clk,\n\
                #input reset,\n\
                #output reg [3:0] q\n\
                #);"
                #print(prompt_test)
        
        #env = LLMQueryEnv(orig_prompt=prompt, orig_module=module_name, file_path=prob_files_path)
        #start_prompt="def hello_world():"
    idx_ep = 0
    #mctsTree = initialize_MCTS_tree(LLMQueryEnv(orig_prompt=start_prompt))
    mctsTree = initialize_MCTS_tree(LLMQueryEnv(orig_prompt=prompt, orig_module=module_name, file_path=prob_files_path))
    print("Episode not stated yet!")
    while idx_ep<num_episodes:
        print("******** EPISODE-{}************".format(idx_ep+1))
        mctsTree = execute_episode(mctsTree,simulation_per_episode)
        evalMctsMaxValue,evalMctsRobustValue = test_episode(mctsTree)
        csvFileHandler.write("{},{}\n".format(evalMctsMaxValue,evalMctsRobustValue))
        idx_ep+=1
