LLMs utilizing MCTS

Requirements:
The program below utilizes a 2-billion parameter LLM. This requires at least 11 GB of storage available for the transformer model parameters to be downloaded and utilized.
Additionally, the model requires up to 24 GB of RAM to generate an output for a given prompt file from the LLM (NVIDIA RTX A5000 24 GB was used).
Setup
Before you run the program below, ensure the following steps are met.
Install:

"pip install -r requirements.txt"
May need to install additional modules related to your individual GPU being utilized. (Current requirements have NVIDIA packages for a RTX A5000).


The following instructions detail the terminal commands to utilize when 
To run the LLM for a given prompt file, follow the following commands below:

python main.py --dumpdir output_files --runID 0 --sim 1 --ep 1 --prompt_path mac/register.v --op greedy --csv counter.csv

`pip install -r requirements.txt`
