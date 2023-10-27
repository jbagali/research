#!/bin/bash

# List of mod_dir and mod_name values
mod_dirs=("VGen-main/prompts-and-testbenches/basic1" "VGen-main/prompts-and-testbenches/basic2" "VGen-main/prompts-and-testbenches/basic3"
            "VGen-main/prompts-and-testbenches/basic4" "VGen-main/prompts-and-testbenches/intermediate1" "VGen-main/prompts-and-testbenches/intermediate2"
            "VGen-main/prompts-and-testbenches/intermediate3" "VGen-main/prompts-and-testbenches/intermediate4" "VGen-main/prompts-and-testbenches/intermediate5"
            "VGen-main/prompts-and-testbenches/intermediate6" "VGen-main/prompts-and-testbenches/intermediate7" "VGen-main/prompts-and-testbenches/intermediate8"
            "VGen-main/prompts-and-testbenches/advanced1" "VGen-main/prompts-and-testbenches/advanced2" "VGen-main/prompts-and-testbenches/advanced3"
            "VGen-main/prompts-and-testbenches/advanced4" "VGen-main/prompts-and-testbenches/advanced5")
mod_names=("prompt3_wire_assign.v" "prompt3_and_gate.v" "prompt3_priority_encoder.v" "prompt3_mux.v" "prompt3_half_adder.v" "prompt3_counter.v" "prompt3_lfsr.v" 
            "prompt3_simple-fsm.v" "prompt3_shift-left-rotate.v" "prompt3_ram.v" "prompt3_permutation.v" "prompt3_truthtable.v" "prompt3_signed-addition-overflow.v"  
       "prompt3_countslow.v" "prompt3_advfsm.v" "prompt3_advshifter.v" "prompt3_abro.v")

# Loop through the arrays and run the command
for i in {1..17}; do
    csv_file="log${i}.csv"
    mod_dir=${mod_dirs[$i - 1]}
    mod_name=${mod_names[$i - 1]}
    
    CUDA_VISIBLE_DEVICES=2 python main.py --dumpdir scripts/dump --runID 0 --sim 200 --ep 1 --mod_dir "${mod_dir}" --mod_name "${mod_name}" --csv "${csv_file}"
done