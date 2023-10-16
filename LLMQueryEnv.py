import numpy as np
import gym # open ai gym
import os,re
import time
import torch
import random
import subprocess
from static_env import StaticEnv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def seed_everything(seed=42):                                                 
    random.seed(seed)                                                     
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)                                                   
        torch.cuda.manual_seed_all(seed)                                             
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device("cuda")
        #print("Using GPU")
else:
    device = torch.device("cpu")
#test
class LLMQueryEnv(gym.Env, StaticEnv):
    """
    Simple gym environment with the goal to navigate the player from its
    starting position to the highest point on a two-dimensional map within
    a limited number of steps. Rewards are defined as the difference in
    altitude between states minus a penalty for each step. The player starts
    in the lower left corner of the map and the highest point is in the upper
    right corner. Map layout mirrors CliffWalking environment:
    top left = (0, 0), top right = (0, m-1), bottom left = (n-1, 0),
    bottom right = (n-1, m-1).
    The setup of this environment was inspired by the energy landscape in
    protein folding.
    """

    # origAIG incomplete prompt
    
    def __init__(self,orig_prompt="def hello_world():", orig_module="hello_world", file_path = ""):
        seed_everything()
        model_name = "shailja/CodeGen_2B_Verilog"
        self.tokenizer = AutoTokenizer.from_pretrained("shailja/fine-tuned-codegen-2B-Verilog")
        self.model = AutoModelForCausalLM.from_pretrained("shailja/fine-tuned-codegen-2B-Verilog").to(device)
        self.orig_prompt = orig_prompt
        self.init_state = self.get_tokenized_state(self.orig_prompt)
        self.num_tokens=0
        self.n_actions = 50295 #self.tokenizer.vocab_size
        #self.stopwords = ['\n\n']
        self.stopwords = ['endmodule']
        self.depth=500
        self.orig_module = orig_module
        self.file_path = file_path
            #self.ep_length = NUM_LENGTH_EPISODES # not required

    def get_tokenized_state(self,prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids.numpy()

    def get_initial_state(self):
        state = self.init_state
        return state

    def reset(self):
        """ Go back to the initial state. """
        state = self.init_state
        return state

    def trim_with_stopwords(self, currentState):
        for w in sorted(self.stopwords, key=len, reverse=True):
            if currentState.endswith(w):
                currentState = currentState[:-len(w)], w
                # print('Trimmed', repr(currentState))
                return currentState


    def isPromptComplete(self,currentState,depth):
        with torch.no_grad():
            torchState = torch.from_numpy(currentState).to(device)
            decoded = self.tokenizer.decode(currentState[0])    
        #print('decoded state',repr(decoded))
        for w in sorted(self.stopwords, key=len, reverse=True):
            if decoded.endswith(w):
                print("Output Verilog from prompt is complete.")
                #Getting resulting Verilog code.
                verilog_code = self.get_prompt_from_state(currentState)
                print(verilog_code)
                # Write the Verilog code to a temporary file - file named after module name.
                output_verilog_file = self.orig_module + ".v"
                with open(output_verilog_file, 'w') as temp_file:
                    temp_file.write(verilog_code)

                #Setting the testbench file path (assuming in same location as prompt file).
                testbench_path = self.file_path + "/tb_" + self.orig_module + ".v"
                #Check compilability.
                compilability = self.compilation_check(testbench_path, output_verilog_file)
                #Call functionality check if compilable.
                if compilability:
                    functionality = self.functionality_check()
                #Cleaning up files.
                #os.remove(output_verilog_file)
                #os.remove('simulation')
                return True

    def getPromptScore(self,currentState=""):
        print("Running getPromptScore: ")
        #Specify your bash script to be utilized here.
        
        #bash_script = "/home/grads/m/matthewdelorenzo/mcts/scripts/synth_gcd.sh"
        #module_dump_folder = "/home/grads/m/matthewdelorenzo/mcts/scripts/dump/" + self.orig_module
        bash_script = "scripts/synth_gcd.sh"
        module_dump_folder = "scripts/dump/" + self.orig_module
        #Creating dump file for results to be placed if not already created.
        if not os.path.exists(module_dump_folder):
            try:
                os.makedirs(module_dump_folder, mode=0o777)
            except OSError as e:
                print("Error creating dump file: ", e)
        #Editing the bash script to run the output verilog module.
        x= self.edit_script_variables(bash_script, self.orig_module, self.orig_module + ".v")
        #Running the script file (with executable permission).
        try:
            subprocess.run(['bash', '-c', f'chmod +x {bash_script} && {bash_script}'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running bash script: {e}")
            exit(1)
        #Retrieving the results from generated log file - specify your filepath.
        logfile_path = module_dump_folder + "/yosys_synth.log"
        #Sleep command to allow new files to be accessed without timing issues.
        #May need more advanced delay system...
        time.sleep(1)
        #Retrieving the area and delay values from log file.
        if os.path.isfile(logfile_path):
            area_value = self.extract_area_from_log(logfile_path)
            delay_value = self.extract_delay_from_log(logfile_path)
            print("Initial area: ", area_value)
            print("Initial delay: ", delay_value)
            #Printing results.
            if(area_value is not None and delay_value is not None):
                print()
                print("Currently displaying area/delay scores for ", self.orig_module, " module.")
                print("Area of the chip design is: ", area_value)
                print("Delay value for the chip design is: ", delay_value)
                print("Score (1/chip area): ", 1 / float(area_value))
                #Currently returning area and delay values.
                return (1 / float(area_value))
            else:
                print("Verilog code has not area or delay value (error in compilation).")
                return 0
        else:
            print("Error in filepath of Yosys sythesis results.")
            return None
    
    def compilation_check(self, testbench_path, module_path):
         # Compile the Verilog code using the iVerilog
        try:
            # Run the Verilog compiler and capture the output and exit code - specify your testbench file to your prompt here.
            compile_output = subprocess.check_output(['iverilog', '-o', 'simulation', testbench_path, module_path], stderr=subprocess.STDOUT)
            compile_exit_code = 0  # Compilation successful
            print("Output Verilog module compiles successfully.")
            return True
        except subprocess.CalledProcessError as e:
            # Compilation failed, get error message
            compile_output = e.output
            compile_exit_code = e.returncode
            print("Verilog compilation failed, error: ", compile_exit_code)
            print("Compilation output: ", compile_output)
            return False

    #Helper function to check the functionality of output Verilog code against its respective testbench.
    def functionality_check(self):
        try:
        #Executing the compiled simulation file.
            simulation_output = subprocess.check_output(['vvp', 'simulation'], stderr=subprocess.STDOUT)
            simulation_exit_code = 0
        except subprocess.CalledProcessError as e:
            simulation_output = e.output
            simulation_exit_code = e.returncode
        #Displaying result cases of simulation.
        print("Simulation output: ", simulation_output)
        if simulation_exit_code == 0:
            print("Verilog testbench simulation ran successfully.")
            if b"all tests passed" in simulation_output:
                print("All testbench tests passed!")
                #self.getPromptScore()
                return True
            else:
                print("Some testbench tests failed.")
                #self.getPromptScore()
                return False
        else: 
            print("Verilog testbench simulation failed.")
            #self.getPromptScore() 
            return False
        
    #Helper - writes to the bash script to run the specific design/verilog file being synthesized.
    def edit_script_variables(self, filename, design_name, file_name):
        with open(filename, "r") as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith("export DESIGN_NAME="):
                lines[i] = f'export DESIGN_NAME={design_name}\n'
            elif line.startswith("export VERILOG_FILE="):
                lines[i] = f'export VERILOG_FILE={file_name}\n'
        with open(filename, "w") as file:
            file.writelines(lines)
        
    #Helper - retrieving the delay value from the Yosys log file.
    def extract_delay_from_log(self, filepath):
        print("retrieving delay (called)")
        print("Delay filepath: ", filepath)
        with open(filepath, "r") as file:
            for line in file:
                if "Delay = " in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Delay":
                            return parts[i + 2]
        print("Delay could not be found in synthesis results.")
        return None
    
    #Helper - retrieving the area value from the Yosys log file.
    def extract_area_from_log(self, filepath):
        #Reading data.
        with open(filepath, 'r') as file:
            content = file.read()
        #Finding chip area number.
        pattern = r'Chip area for module .+: (\d+\.\d+)'
        match = re.search(pattern, content)
        if match:
            chip_area = match.group(1)
            return chip_area
        else:
            print("Error: chip area not found in synthesis results.")
            return None

    def next_state(self,state,action):
        nextState = np.append(state,np.array([[action]]),axis=-1)
        return nextState

    def is_done_state(self,state,depth):
        if self.isPromptComplete(state,depth):
            return True
        elif depth>=self.depth:
            return True
        else:
            return False

    def get_prompt_from_state(self,state):
        with torch.no_grad():
            torchState = torch.from_numpy(state).to(device)
            prompt_from_state = self.tokenizer.decode(torchState[0])
            return prompt_from_state

    def getLLMestimates(self,state):
        with torch.no_grad():
            torchState = torch.from_numpy(state).to(device)
            output = self.model(input_ids=torchState)
            next_token_logits = output.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            #print("next_token logits: ", len(next_token_logits))
            return next_token_probs.detach().cpu().numpy()

    def get_best_terminal_state(self,state,depth):
        with torch.no_grad():
            torchState = torch.from_numpy(state).to(device)
            while not self.is_done_state(state,depth):
                #Could call getLLMEstimates?
                output = self.model(input_ids=torchState)
                next_token_logits = output.logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
                torchState = torch.cat([torchState, sorted_ids[None,0, None]], dim=-1)
                state = torchState.detach().cpu().numpy()
                depth+=1
            return state

    def get_montecarlo_return(self,state,depth):
        best_terminal_state = self.get_best_terminal_state(state,depth)
        complete_prompt = self.get_prompt_from_state(best_terminal_state)
        filteredGen = self.trim_with_stopwords(complete_prompt)
        score = self.getPromptScore(filteredGen)
        #score = self.getPromptScore(complete_prompt)
        return score

    def get_return(self,state,depth):
        ##Sanity Check##
        if not self.is_done_state(state,depth):
            print("Serious error")
            exit(1)
        complete_prompt = self.get_prompt_from_state(state)
        #score = self.getPromptScore(complete_prompt)
        score = self.getPromptScore(state)
        return score
        

if __name__ == '__main__':
    #---------------------CONFIGURE the below file path/name to your module to test-------------------
    #Configure your path to the prompts/testbenches folder you want to test.
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
    
    env = LLMQueryEnv(orig_prompt=prompt, orig_module=module_name, file_path=prob_files_path)
    print("env created")
    init_state = env.get_initial_state()
    depth=0
    print("Main: ", init_state)
    ### Rollout return ###
    finalState = env.get_best_terminal_state(init_state,depth)
    promptGen = env.get_prompt_from_state(finalState)
    filteredGen=env.trim_with_stopwords(promptGen)
    print('Filtered relevant code with stop words {}-->\n{}\n'.format(env.stopwords, filteredGen))
    
    #### Get next best state ###
    best_prediction = np.argmax(env.getLLMestimates(init_state))
    next_state = env.next_state(init_state,best_prediction)
    prompt = env.get_prompt_from_state(next_state)
    depth+=1
    print('1',prompt)
    #### Again, get the next best state ###
    best_prediction = np.argmax(env.getLLMestimates(next_state))
    next_state = env.next_state(next_state,best_prediction)
    prompt = env.get_prompt_from_state(next_state)
    depth+=1
    print('2',prompt)

    
