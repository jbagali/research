"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
#import imp
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
from torch.nn.parallel import DistributedDataParallel
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
    #parser.add_argument('--init_prompt', type=str, required=False,
    #                    help='Initial prompt')
    parser.add_argument('--dumpdir', type=str, required=True,default="",
                        help='DUMP directory for storing output files.')
    parser.add_argument('--op', type=str, required=True,default="",
                        help = "Please state which operation to perform: 'mcts', 'beam', or 'greedy'.")
    parser.add_argument('--runID', type=int, required=True, default=0,
                        help='Run ID')
    parser.add_argument('--sims', type=int, required=True, default=1000,
                        help='Simulations per MCTS episode')
    parser.add_argument('--ep', type=int, required=True, default=50,
                        help='#MCTS episode')
    parser.add_argument('--pr', type=int, required=False, default=1,
                        help='#processes executing MCTS')
    #parser.add_argument('--mod_dir', type=str, required=False, default = "VGen-main/prompts-and-testbenches/intermediate1",
    #                    help="Directory of prompt files (ex: intermediate2)")
    #parser.add_argument('--mod_name', type=str, required=False, default = "prompt3_half_adder.v",
    #                    help = "Name of prompt file (ex: prompt1_counter.v)")
    parser.add_argument('--csv', type=str, required=False, default = "log.csv",
                        help = "Name of csv file. (ex: log.csv)")
    parser.add_argument('--prompt_path', type=str, required=True, default = "",
                        help = "Filepath of prompt file from current directory (ex: filepath/prompt1_counter.v)")
    parser.add_argument('--tb_path', type=str, required=True, default = "",
                        help = "Filepath of counter filepath from current directory (ex: filepath/prompt1_counter.v)")
    parser.add_argument('--module_name', type=str, required=True, default = "",
                        help = "Name of module in prompt for which the LLM will finish creating. Ex: counter")

    args = parser.parse_args()
    
    #origPrompt = args.init_prompt
    rootDumpDir = args.dumpdir
    runID = args.runID
    simulation_per_episode = args.sims
    num_episodes = args.ep
    num_processes = args.pr
    operation = args.op
    #file_dir = args.mod_dir
    #file_name = args.mod_name
    csv_name = args.csv
    csv_logger = CsvLogger(csv_name)
    row_data = {}

    prompt_filepath = args.prompt_path
    tb_filepath = args.tb_path
    module_name = args.module_name
    
    prompt_str = ""
    #set_start_method("spawn")

    #runFolder = osp.join(rootDumpDir,'run'+str(runID))
    #csvResultFile = osp.join(rootDumpDir,'data_run'+str(runID)+".csv") # Data to store area obtained in each iteration and best so far
    #logFilePath = osp.join(runFolder,'log_run'+str(runID)+".log")   # Log data to dump area and read
    
    if not osp.exists(rootDumpDir):
        os.mkdirs(rootDumpDir, mode=0o777 )
        
    #if not osp.exists(runFolder):    
    #    os.mkdir(runFolder)

    #csvFileHandler = open(csvResultFile,'w')
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        #print("Using GPU")
    else:
        device = torch.device("cpu")


    #promptfile_path = file_dir + "/" + file_name
    #Reading input file data and finding the module name.
    #Parsing the file name for the problem name (counter, lfsr, etc).
    # file_name_parts = file_name.split("_", 1)
    # if len(file_name_parts) > 1:
    #     problem_name = "_".join(file_name_parts[1:])
    #     print("Problem: ", problem_name)
    #     print("Joined problem name: ", problem_name)
    #     if problem_name.endswith(".v"):
    #             problem_name = problem_name.replace(".v","")
    # else:
    #     print("Error parsing file name - check formatting (ex: prompt1_counter.v)")
    #     exit(1)

    print(prompt_filepath)
    try:
        with open(prompt_filepath, "r") as prompt_file:
            prompt_str = prompt_file.read()
            print("Prompt str: ", prompt_str)
    except:
        print("Main: Error reading prompt file.")
        exit(1)
        

    print("Loading LLM model...")
    model_name = "shailja/CodeGen_2B_Verilog"
    tokenizer = AutoTokenizer.from_pretrained("shailja/fine-tuned-codegen-2B-Verilog")
    model = AutoModelForCausalLM.from_pretrained("shailja/fine-tuned-codegen-2B-Verilog").to(device)
    #model = DistributedDataParallel(model, device_ids=[0,1,2], find_unused_parameters=True)
    print("Initializing MCTS tree/LLM env...")
    idx_ep = 0
    
    print("Episode not stated yet!")
    print("Simulations per episode: ", simulation_per_episode)
    
    
    while idx_ep<num_episodes:
        print("******** EPISODE-{}************".format(idx_ep+1))
        if(operation == "mcts"):
            print("ORIG MODILE: ", module_name)
            merged_tree = initialize_MCTS_tree(LLMQueryEnv(csv_logger, row_data, orig_prompt=prompt_str, op = operation, orig_module=module_name, file_path=prompt_filepath, tb_path = tb_filepath, dump_path = rootDumpDir,
                                                        model_name=model_name, tokenizer=tokenizer, model=model))
            merged_tree = execute_episode(merged_tree,simulation_per_episode)
            print("ROBUST FINAL VALUE:")
            evalMctsRobustValue, evalMctsMaxValue = test_episode(merged_tree)
            ##csvFileHandler.write("{},{}\n".format(evalMctsRobustValue))

        elif (operation == "beam"):
            env = LLMQueryEnv(csv_logger, row_data, orig_prompt=prompt_str, op = operation, orig_module=module_name, file_path=prompt_filepath, tb_path = tb_filepath, dump_path = rootDumpDir,
                                                        model_name=model_name, tokenizer=tokenizer, model=model)
            for i in range(simulation_per_episode):
                env.row_data['episode'] = idx_ep
                env.row_data['currentRun'] = i
                init_state = env.get_initial_state()
                output = env.beam_search(init_state)
                
        elif (operation == "greedy"):
            for i in range(simulation_per_episode):
                print("----GREEDY LLM OUTPUT - ITERATION: ", i, " ----")
                print("---------------")
                env = LLMQueryEnv(csv_logger, row_data, orig_prompt=prompt_str, op = operation, orig_module=module_name, file_path=prompt_filepath, tb_path = "", dump_path = rootDumpDir,
                                                            model_name=model_name, tokenizer=tokenizer, model=model)
                env.row_data['episode'] = idx_ep
                env.row_data['currentRun'] = i
                init_state = env.get_initial_state()
                finalState = env.get_best_terminal_state(init_state,0)
                promptGen = env.get_prompt_from_state(finalState)
                filteredGen=env.trim_with_stopwords(promptGen)
                score = env.getPromptScore(filteredGen)
                #print('Filtered relevant code with stop words {}-->\n{}\n'.format(env.stopwords, filteredGen))
                #print("Error please correct the operation type.")
            break
        else:
            print("Error reading --op parameter. Please only state 'beam', 'greedy', or 'mcts' as your input.")
            exit(1)

        idx_ep+=1

    #print("Num simulations: ", merged_tree.num_simulations)
    



    
