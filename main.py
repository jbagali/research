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
from concurrent.futures import ThreadPoolExecutor,as_completed
import multiprocessing
from multiprocessing import set_start_method
from LLMQueryEnv import LLMQueryEnv
from mcts import execute_episode,test_episode,initialize_MCTS_tree,merge_trees
import argparse,os,re
import os.path as osp
import torch,shutil
import statistics,pickle
import csv
from mcts import MCTS, initialize_thread_tree
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class CsvLogger:
    def __init__(self, filename):
        self.filename = filename
        # Initialize the CSV file with the header
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Area", "Delay", "Score", "Current Run", "Episode", "Verilog"])  # Add header row

    def log(self, data):
        # 'data' is expected to be a dictionary like:
        # {'area': 123.4, 'delay': 5.67, 'score': 8.9, 'currentRun': 10, 'episode': 1}
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            # Make sure the order of keys matches the order of your CSV columns
            writer.writerow([data['area'], data['delay'], data['score'], data['currentRun'], data['episode'], data['verilog']])




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
    parser.add_argument('--pr', type=int, required=False, default=1,
                        help='#processes executing MCTS')
    parser.add_argument('--mod_dir', type=str, required=False, default = "VGen-main/prompts-and-testbenches/intermediate1",
                        help="Directory of prompt files (ex: intermediate2)")
    parser.add_argument('--mod_name', type=str, required=False, default = "prompt3_half_adder.v",
                        help = "Name of prompt file (ex: prompt1_counter.v)")
    parser.add_argument('--csv', type=str, required=False, default = "log.csv",
                        help = "Name of csv file. (ex: log.csv)")

    args = parser.parse_args()
    
    origPrompt = args.init_prompt
    rootDumpDir = args.dumpdir
    runID = args.runID
    simulation_per_episode = args.sims
    num_episodes = args.ep
    num_processes = args.pr
    file_dir = args.mod_dir
    file_name = args.mod_name
    csv_name = args.csv
    csv_logger = CsvLogger(csv_name)
    row_data = {}
    
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
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        #print("Using GPU")
    else:
        device = torch.device("cpu")


    if not origPrompt:
        #-----------Alter the two lines below for your prompt files.---------------#
        #TODO
        #------For now, file_name (without the .v) should be the same as module name in prompt file for code to run.

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
        

    print("Loading LLM model...")
    model_name = "shailja/CodeGen_2B_Verilog"
    tokenizer = AutoTokenizer.from_pretrained("shailja/fine-tuned-codegen-2B-Verilog")
    model = AutoModelForCausalLM.from_pretrained("shailja/fine-tuned-codegen-2B-Verilog").to(device)
    
    idx_ep = 0
    merged_tree = initialize_MCTS_tree(LLMQueryEnv(csv_logger, row_data, orig_prompt=prompt_str, orig_module=problem_name, file_path=file_dir,
                                                    model_name=model_name, tokenizer=tokenizer, model=model))
    
    print("Episode not stated yet!")
    print("Simulations per episode: ", simulation_per_episode)
    
    
    while idx_ep<num_episodes:
        print("******** EPISODE-{}************".format(idx_ep+1))
        merged_tree = execute_episode(merged_tree,simulation_per_episode)
        evalMctsMaxValue,evalMctsRobustValue = test_episode(merged_tree)
        csvFileHandler.write("{},{}\n".format(evalMctsMaxValue,evalMctsRobustValue))
        idx_ep+=1

    print("Num simulations: ", merged_tree.num_simulations)
    
    
    # while idx_ep<num_episodes:
    #     with ThreadPoolExecutor(max_workers=num_processes) as executor:
    #         thread_trees = list(executor.map(
    #             initialize_thread_tree, 
    #             range(num_processes),
    #             [prompt_str] * num_processes,
    #             [problem_name] * num_processes,
    #             [file_dir] * num_processes,
    #             [model_name] * num_processes,
    #             [tokenizer] * num_processes,
    #             [model] * num_processes
    #             ))
    #         futures = [executor.submit(execute_episode, tree, simulation_per_episode) for tree in thread_trees]
    #         results = []
    #         for future in as_completed(futures):
    #             result_tree = future.result()
    #             results.append(result_tree)
    #        #results = [future.result() for future in futures]
    #     idx_ep+=1  
    
    # print("# thread trees: ", len(results))
    # print("Calling merge trees...")
    # merge_trees(merged_tree, results)





    #test
    #merge
    #while idx_ep<num_episodes:
    #    with ProcessPoolExecutor(max_workers=simulation_per_episode) as executor:
   #         futures = [executor.submit(execute_episode, mctsTree, 1) for _ in range(simulation_per_episode)]
    #        #results = [future.result() for future in futures]
    #    idx_ep+=1  
    
    #while idx_ep<num_episodes:
    #    with ProcessPoolExecutor(max_workers=simulation_per_episode) as executor:
    #        futures = [executor.submit(execute_episode, mctsTree, 1) for _ in range(simulation_per_episode)]
    #        #results = [future.result() for future in futures]
    #   idx_ep+=1  
    
   # print(mctsTree.num_simulations)
    
    

    
