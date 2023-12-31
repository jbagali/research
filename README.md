### LLM with MCTS - README

This file describes the requirements, setup, and commands required to run the LLM with Monte Carlo Tree-Search (MCTS) and other operations.

### Requirements:
The program below utilizes a 2-billion parameter LLM. This requires at least 11 GB of storage available for the transformer model parameters to be downloaded and utilized.
Additionally, the model requires up to 24 GB of RAM to generate an output for a given prompt file from the LLM (NVIDIA RTX A5000 24 GB was used).

### Setup:
Clone the repo above such that the directories within the repositories are of the same structure. <br />
Then, before you run the program below, ensure the following steps are met.

Install: "pip install -r requirements.txt"
May need to install additional modules related to your individual GPU being utilized (current requirements.txt includes required packages for a NVIDIA RTX A5000).

### Run:
The following instructions detail the terminal command parameters that can be utilized to
to generate an LLM output for a given prompt file. This includes generating results for basic LLM results (greedy and beam search), as well as MCTS.
Some additional parameters are described below.

An example terminal command is the following:<br />
python main.py --dumpdir output_files --sim 1 --ep 1 --prompt_path prompt_tb_files/mac/mac_4.v --tb_path prompt_tb_files/mac/tb_mac_4.v --module_name mac_4 --op mcts --csv mac_new.csv

### Below are the terminal command hyperparameters and their descriptions.
```
--dump_dir:    Filepath of the directory in which output files are stored.
--op:          Specifies which LLM operation to perform ('mcts', 'beam', or 'greedy').
--sims:        Simulations per MCTS episode.
--csv: 	       Name of CSV file to which results will be stored.
--prompt_path: Filepath of prompt file (.v) from current directory. (ex: filepath/prompt1_counter.v)
--tb_path:     Filepath of corresponding testbench file (to the prompt file) from current directory (ex: filepath/prompt1_counter.v).
--module_name: Name of module in prompt for which the LLM will finish generating. (Ex: counter)
```
The files within the directory "prompt_tb_files" contain the prompts and testbenches that can be utilized with the commands above for Verilog adder, multiplier and Multiply Accumulate Unit (MAC) designs.
