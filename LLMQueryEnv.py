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

def seed_everything(operation):  
    if(operation == "beam" or operation == "greedy"):
        seed = random.randint(1, 1000000)     
        random.seed(seed)                                                     
        torch.manual_seed(seed)
        np.random.seed(seed)
        print("Env seed: ", seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)                                                   
            torch.cuda.manual_seed_all(seed)                                             
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False
    elif(operation == "mcts"):
        seed = 42                                          
        random.seed(seed)                                                     
        torch.manual_seed(seed)
        np.random.seed(seed)
        print("Env seed: ", seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)                                                   
            torch.cuda.manual_seed_all(seed)                                             
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        print("Error in --op parameter. Please state 'mcts', 'greedy', or 'beam' as your decision.")

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
    
    def __init__(self, csv_logger=None, row_data=None, op = "mcts", orig_prompt="def hello_world():", orig_module="hello_world", file_path = "", tb_path = "", dump_path = "",
                 model_name=None, tokenizer=None, model=None):
        self.op = op
        seed_everything(self.op)

        self.csv_logger = csv_logger
        self.row_data = row_data
        model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.orig_prompt = orig_prompt
        self.init_state = self.get_tokenized_state(self.orig_prompt)
        self.num_tokens=0
        self.n_actions = 10 #self.tokenizer.vocab_size
        self.stopwords = ['endmodule']
        #Limit to token generation before cutoff.
        self.depth=1000
        self.orig_module = orig_module
        self.prompt_path = file_path
        self.tb_path = tb_path
        self.dumppath = dump_path
        self.non_compilable_attempts = 0
        compilation_output = None
        self.compilable = False
        self.functional = False
        self.first_successful_product = None
        self.beam_count = 0
        #self.top_token_ids = None
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
        #start_time = datetime.now()
        with torch.no_grad():
            torchState = torch.from_numpy(currentState).to(device)
            decoded = self.tokenizer.decode(currentState[0])    
        #print('decoded state',repr(decoded))
        for w in sorted(self.stopwords, key=len, reverse=True):
            if decoded.endswith(w):
                #if(self.tb_path = ""):  
                self.verilogFunctionalityCheck(currentState)
                if self.compilable:     #if compilable, finish prompt generation.
                    return True
                #elif self.non_compilable_attempts >= 2:    #if continued generation of module 2+ times, finish.
                #    return True
                elif b'Unknown module type' in self.compilation_output:    #if unknown module, continue generation.
                    #self.non_compilable_attempts += 1
                    return False
                else:   #if compilation error is of origin other than "undefined module", finish generation.
                    return True

                #return True
            else:
                return False
            
    def verilogFunctionalityCheck(self, currentState):
        verilog_code = self.get_prompt_from_state(currentState)
        # Write the Verilog code to a temporary file - filenamed after module name.
        print(verilog_code)
        module_dump_folder = self.dumppath + "/" + str(os.getpid()) + "_" + self.orig_module
        if not os.path.exists(module_dump_folder):
            try:
                os.makedirs(module_dump_folder, mode=0o777)
            except OSError as e:
                print("Error creating dump file: ", e)

        output_verilog_file = str(os.getpid()) + "_" + self.orig_module + ".v"       

        output_file_path = os.path.join(module_dump_folder, output_verilog_file)
        #try:
        #    module_name = "module " + self.orig_module
        #    #print("MODULE NAME: ", module_name)
        #    module_index = verilog_code.find(module_name)
        #    verilog_code = verilog_code[module_index:]
        #except:
        #    print("Prompt does not contain the term 'module' - please readjust.")   

        with open(output_file_path, 'w') as temp_file:
            temp_file.write(verilog_code)
        #TMP
        #if(self.tb_path != ""):
        self.row_data['verilog'] = verilog_code

        os.chmod(output_file_path, 0o777)
        #Setting the testbench file path (assuming in same location as prompt file).

        
        
        #TMP EDIT
        self.compilable = self.compilation_check(output_file_path, self.tb_path)
        #Call functionality check if compilable.
        
        
        if self.compilable:
            self.functional = self.functionality_check()
        
        return 0

    def getPromptScore(self, currentState=""):
        print("Running getPromptScore: ")

        if not self.compilable:
            self.row_data['area'] = 'N/A'
            self.row_data['delay'] = 'N/A'
            self.row_data['score'] = -1
            return -1
        
        #TMP EDIT!
        if not self.functional:
            self.row_data['area'] = 'N/A'
            self.row_data['delay'] = 'N/A'
            self.row_data['score'] = -.1
            return  -.1
        
        #TMP EDIT
        #Specify your bash script to be utilized here.
        bash_script = "scripts/synth_gcd.sh"

        module_dump_folder = self.dumppath + "/" + str(os.getpid()) + "_" + self.orig_module
        #print("Dump folder: ", module_dump_folder)

        #Creating dump file for results to be placed if not already created.
        if not os.path.exists(module_dump_folder):
            try:
                os.makedirs(module_dump_folder, mode=0o777)
            except OSError as e:
                print("Error creating dump file: ", e)
        #Editing the bash script to run the output verilog module.
        new_script_path = os.path.join(module_dump_folder, "synth_script.sh")
        print(self.orig_module)
        print(new_script_path)
        shutil.copy(bash_script, new_script_path)
        x= self.edit_script_variables(new_script_path, self.orig_module, str(os.getpid()) + '_' + self.orig_module + ".v")
        #Running the script file (with executable permission).
        try:
            start_time = datetime.now()
            subprocess.run(['bash', '-c', f'chmod +x {new_script_path} && {new_script_path}'], check=True)
            end_time = datetime.now()
            time_difference = end_time - start_time
            seconds = time_difference.total_seconds()
            print("Running bash in x seconds: ", seconds)
        except subprocess.CalledProcessError as e:
            print(f"Error running bash script: {e}")

        #Retrieving the results from generated log file - specify your filepath.
        logfile_path = module_dump_folder + "/yosys_synth.log"
        if os.path.isfile(logfile_path):
            area_value = self.extract_area_from_log(logfile_path)
            delay_value = self.extract_delay_from_log(logfile_path)
            if(self.functional and area_value is not None and delay_value is not None):
                area_value = float(area_value)
                delay_value = float(delay_value)
                current_area_delay_product = area_value * delay_value
                if(self.first_successful_product == None):
                    self.first_successful_product = current_area_delay_product
                
                reward = .1 + (1 - (current_area_delay_product / self.first_successful_product))
              
                    #if current_area_delay_product == self.first_successful_product:
                    #    reward = 1
                    #else:
                    #    reward = 1 + (((current_area_delay_product - self.first_successful_product) / self.first_successful_product) * -1)
                print()
                print("Currently displaying area/delay scores for ", self.orig_module, " module.")
                print("Area of the chip design is: ", area_value)
                print("Delay value for the chip design is: ", delay_value)
                print("Product: ", current_area_delay_product)
                print("Score (1/chip area): ", reward)
                self.row_data['area'] = area_value
                self.row_data['delay'] = delay_value
                self.row_data['score'] = reward
                             #Currently returning area and delay values.
                #try:
                #    print("Removing dump files...")
                #    os.remove(logfile_path)
                #except OSError as e:
                #    print(f"Error deleting log file: {e}")

                return reward
            else:
                print("Error retrieving area/delay from results.")
                self.row_data['area'] ='N/A'
                self.row_data['delay'] = 'N/A'
                self.row_data['score'] = -.75
                return -.75
        else:
            print("Filepath of Yosys results not recognized.")
            return None
    
    def compilation_check(self, module_path, testbench_path=None):
         # Compile the Verilog code using the iVerilog
        try:
            print("Path: ", os.path.join(self.dumppath + "/" + str(os.getpid()) + "_" + self.orig_module, str(os.getpid()) + '_simulation'))
            compile_output = subprocess.check_output(['iverilog', '-o', os.path.join(self.dumppath + "/" + str(os.getpid()) + "_" + self.orig_module, str(os.getpid()) + '_simulation'), testbench_path, module_path], stderr=subprocess.STDOUT)
            compile_exit_code = 0  # Compilation successful
            self.compilation_output = None
            print("Output Verilog module compiles successfully.")
            return True
        except subprocess.CalledProcessError as e:
            # Compilation failed, get error message
            compile_output = e.output
            compile_exit_code = e.returncode
            print("Verilog compilation failed, error: ", compile_exit_code)
            #print("Compilation output: ", compile_output)
            self.compilation_output = compile_output
            return False

    #Helper function to check the functionality of output Verilog code against its respective testbench.
    def functionality_check(self):
        try:
            sim_path = os.path.join(self.dumppath + "/" + str(os.getpid()) + "_" + self.orig_module, str(os.getpid()) + '_simulation')
            simulation_output = subprocess.check_output(['vvp', sim_path], stderr=subprocess.STDOUT)
            simulation_exit_code = 0
        except subprocess.CalledProcessError as e:
            simulation_output = e.output
            simulation_exit_code = e.returncode

        if simulation_exit_code == 0:
            print("Verilog testbench simulation ran successfully.")
            if b"all tests passed" in simulation_output or b"All tests passed" in simulation_output:
                print("Simulation output: ", simulation_output)
                print("All testbench tests passed!")
                return True
            else:
                print("Some testbench tests failed.")
                print("Simulation output: ", simulation_output)
                return False
        else: 
            print("Verilog testbench simulation failed.")
            print("Simulation output: ", simulation_output)
            return False
        
    #Helper - writes to the bash script to run the specific design/verilog file being synthesized.
    def edit_script_variables(self, filename, design_name, file_name):
        with open(filename, "r") as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith("export ROOT_DIR="):
                lines[i] = f'export ROOT_DIR={os.getcwd()}\n'
            if line.startswith("export MODULE_DIR="):
                lines[i] = f'export MODULE_DIR={"${ROOT_DIR}" + "/" + self.dumppath + "/" + str(os.getpid()) + "_" + self.orig_module}\n'
            if line.startswith("export DESIGN_NAME="):
                #lines[i] = f'export DESIGN_NAME={design_name}\n'
                lines[i] = f'export DESIGN_NAME={design_name}\n'
            elif line.startswith("export VERILOG_FILE="):
                lines[i] = f'export VERILOG_FILE={"${MODULE_DIR}" + "/" + file_name}\n'
            elif line.startswith("export OUTPUT_NAME="):
                lines[i] = f'export OUTPUT_NAME={"${MODULE_DIR}" + "/" + file_name}\n'
            elif line.startswith("export DUMP_DIR="):
                lines[i] = f'export DUMP_DIR={self.dumppath + "/" + str(os.getpid()) + "_" + self.orig_module}\n'
        with open(filename, "w") as file:
            file.writelines(lines)
        
    #Helper - retrieving the delay value from the Yosys log file.
    def extract_delay_from_log(self, filepath):
        #print("retrieving delay (called)")
        #print("Delay filepath: ", filepath)
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
        with open(filepath, "r") as file:
            for line in file:
                if "Chip area for module" in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        chip_area = parts[-1].strip()
                        return chip_area
        print("Error: Chip area ont found in syntheis results.")
        return None

    def next_state(self,state,action):
        #token_id = self.top_token_ids[action]
        nextState = np.append(state,np.array([[action]]),axis=-1)
        return nextState

    def is_done_state(self,state,depth):
        if self.isPromptComplete(state,depth):
            return True
        if depth>=self.depth:
            self.verilogFunctionalityCheck(state)
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
            sorted_probs, sorted_ids = torch.sort(next_token_probs, dim=-1, descending=True)

            sorted_ids_arr = sorted_ids[:].detach().cpu().numpy()
            sorted_probs_arr = sorted_probs[:].detach().cpu().numpy()
            non_comment_mask = np.array([not self.is_comment(token_id) for token_id in sorted_ids_arr])

            
            non_comment_all_ids = sorted_ids_arr[non_comment_mask]
            print("Len original: ", len(sorted_ids_arr), " Len new: ", len(non_comment_all_ids))
            # Apply the mask to get non-comment IDs and their probabilities
            non_comment_ids = non_comment_all_ids[:self.n_actions]
            non_comment_probs = sorted_probs_arr[non_comment_mask][:self.n_actions]
           
            decoded_tokens = [self.tokenizer.decode([prob]) for prob in non_comment_all_ids]
            
            #top_ids = sorted_ids[:self.n_actions].detach().cpu().numpy()
            #top_probs = sorted_probs[:self.n_actions].detach().cpu().numpy() # Get the top 5 probabilities
            #ecoded_tokens = [self.tokenizer.decode([sorted_id]) for sorted_id in sorted_ids]
            
            #print("Probabilities: ", non_comment_probs)
            #print("Ids: ", non_comment_ids)
            #print("All decoded: ", decoded_tokens)
            
            return non_comment_probs, non_comment_ids
    
    def check_sequence_in_ids(self, ids, target_sequence):
        instances = 0
        id_list = ids[0].tolist()
        if len(id_list) < len(target_sequence):
            return False
        for i in range(len(id_list) - len(target_sequence) + 1):
            if id_list[i:i+len(target_sequence)] == target_sequence:
                instances += 1
                if(instances > self.beam_count):
                    self.beam_count += 1
                
                    print("BEAM SEARCH: ID TYPE: ", type(ids))
                    self.verilogFunctionalityCheck(ids.detach().cpu().numpy())
                    if self.compilable:     #if compilable, finish prompt generation.
                        return True
                    #elif self.non_compilable_attempts >= 2:    #if continued generation of module 2+ times, finish.
                    #    return True
                    elif b'Unknown module type' in self.compilation_output:    #if unknown module, continue generation.
                        #self.non_compilable_attempts += 1
                        return False
                    else:   #if compilation error is of origin other than "undefined module", finish generation.
                        return True
                #return True
        return False
    
    def beam_search(self, state):
        #State is self.init_state
        self.beam_count = 0
        target_sequence = [
        self.tokenizer.convert_tokens_to_ids("end"),
        self.tokenizer.convert_tokens_to_ids("module")
        ]
        intermediate_states = []
        def custom_stopping_criterion(input_ids, state):
            return self.check_sequence_in_ids(input_ids, target_sequence)

        torchState = torch.from_numpy(state).to(device)
        beam_output = self.model.generate(
            input_ids = torchState,
            max_length=torchState.size(1) + self.depth,
            num_beams=3,

            stopping_criteria=[custom_stopping_criterion],
            return_full_text=True,

        )
        best_beam_output = beam_output[0]
        best_beam_output_np = best_beam_output.cpu().numpy()
        decoded_tokens = self.tokenizer.decode(best_beam_output_np, skip_special_tokens=True)
        decoded_tokens = decoded_tokens.rstrip("#")
        self.getPromptScore()
        print("Beam token ids: ", best_beam_output_np.tolist())
        print("Beam results: ", decoded_tokens)
        self.csv_logger.log(self.row_data)
        return True


    def is_comment(self, token_id):
        # Implement a function to check if the token is a comment
        decoded_token = self.tokenizer.decode([token_id])
        return '//' in decoded_token or '/*' in decoded_token or '*/' in decoded_token
    
    def get_best_terminal_state(self,state,depth):
        start_time = datetime.now()
        i = 0
        self.non_compilable_attempts = 0
        with torch.no_grad():
            torchState = torch.from_numpy(state).to(device)
            while not self.is_done_state(state,depth):
                #print("Token: ", i)
                #context_state = torchState[:,-500:]
                output = self.model(input_ids=torchState)

                next_token_logits = output.logits[0, -1, :]
                next_token_prob = torch.softmax(next_token_logits, dim=-1)
                #max_token_index = torch.argmax(next_token_prob, dim=-1)
                #selected_token = max_token_index.unsqueeze(0).unsqueeze(0)

                sorted_probs, sorted_ids = torch.sort(next_token_prob, dim=-1, descending=True)
                selected_token = sorted_ids[0].unsqueeze(0).unsqueeze(0)
                chosen_id = selected_token
                chosen_index = 0
                
                if(self.op == "mcts"):
                    while (self.is_comment(chosen_id.item()) and chosen_index < sorted_ids.size(-1)):
                        #
                        # print("Comment: ", chosen_id.item())
                        #print("Incrementing: ", chosen_index)
                        chosen_index += 1
                        chosen_id = sorted_ids[chosen_index].unsqueeze(0).unsqueeze(0)
                    #print(chosen_id.item())

                selected_token = chosen_id

                #print("Good: ", self.tokenizer.decode(selected_token.item()))
                #print("return in: ", seconds, " seconds")
                
                #sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
                #print("Sorted Id: ", sorted_ids[None,0, None])
                #torchState = torch.cat([torchState, sorted_ids[None,0, None]], dim=-1)


                #torchState = torch.cat([torchState, torch.tensor([chosen_id], device=torchState.device).unsqueeze(0)], dim=-1)
                torchState = torch.cat([torchState, selected_token], dim=-1)
                state = torchState.detach().cpu().numpy()
                decoded = self.tokenizer.decode(state[0])    
                depth+=1
                i += 1
            print("Tokens: ", i)
            end_time = datetime.now()
            time_difference = end_time - start_time
            seconds = time_difference.total_seconds()
            print("LLM generates return in: ", seconds, " seconds")
            return state

    def get_montecarlo_return(self,state,depth):
        best_terminal_state = self.get_best_terminal_state(state,depth)
        complete_prompt = self.get_prompt_from_state(best_terminal_state)
        filteredGen = self.trim_with_stopwords(complete_prompt)
        score = self.getPromptScore(filteredGen)
        return score

    def get_return(self,state,depth):
        ##Sanity Check##
        if not self.is_done_state(state,depth):
            print("Serious error")
            exit(1)
        complete_prompt = self.get_prompt_from_state(state)
        score = self.getPromptScore(state)
        return score


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MCTS+LLM')
    parser.add_argument('--sims', type=int, required=True, default=1000,
                        help='Simulations per MCTS episode')
    parser.add_argument('--dumpdir', type=str, required=True,default="",
                        help='DUMP directory for storing output files.')
    parser.add_argument('--prompt_path', type=str, required=False, default = "VGen-main/prompts-and-testbenches/intermediate1/prompt1_counter.v",
                        help="Directory of prompt files (ex: intermediate2)")
    parser.add_argument('--module_name', type=str, required=False, default = "prompt3_half_adder",
                        help = "Name of prompt file (ex: prompt1_counter.v)")
    
    
    args = parser.parse_args()
    promptfile_path = args.prompt_path
    module_name = args.module_name
    sims = args.sims
    dumpdir = args.dumpdir

    #Specify your prompt file to use as input.
    
    try:
        with open(promptfile_path, "r") as prompt_file:
            prompt_str = prompt_file.read()
            print("Prompt str: ", prompt_str)
    except:
        print("Error reading prompt file.")
        exit(1)


    model_name = "shailja/CodeGen_2B_Verilog"
    tokenizer = AutoTokenizer.from_pretrained("shailja/fine-tuned-codegen-2B-Verilog")
    model = AutoModelForCausalLM.from_pretrained("shailja/fine-tuned-codegen-2B-Verilog").to(device)


    print("env created")
    start_time = datetime.now()
    for i in range(sims):
        print("ITERATION: ", i)
        print("---------------")
        env = LLMQueryEnv(csv_logger=None, row_data=None, orig_prompt=prompt_str, orig_module=module_name, file_path=promptfile_path, tb_path = "", dump_path = dumpdir,
                                                     model_name=model_name, tokenizer=tokenizer, model=model)
        init_state = env.get_initial_state()
        finalState = env.get_best_terminal_state(init_state,0)
        promptGen = env.get_prompt_from_state(finalState)
        filteredGen=env.trim_with_stopwords(promptGen)
        print('Filtered relevant code with stop words {}-->\n{}\n'.format(env.stopwords, filteredGen))
    
    # end_time = datetime.now()
    # time_difference = end_time - start_time
    # print("Final end time: ", time_difference)
    
    #--End---
    
    
    # #### Get next best state ###
    # best_prediction = np.argmax(env.getLLMestimates(init_state))
    # next_state = env.next_state(init_state,best_prediction)
    # prompt = env.get_prompt_from_state(next_state)
    # depth+=1
    # print('1',prompt)
    # #### Again, get the next best state ###
    # best_prediction = np.argmax(env.getLLMestimates(next_state))
    # next_state = env.next_state(next_state,best_prediction)
    # prompt = env.get_prompt_from_state(next_state)
    # depth+=1
    # print('2',prompt)

    